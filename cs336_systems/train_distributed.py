
from cs336_systems.ddp import DDPParameter
from cs336_systems.shard_optimizer import ShardOptimizer
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import numpy as np
import math
import typing
import cs336_basics.model as model_module
from cs336_basics.model import Transformer, MultiHeadSelfAttention
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train import AdamW, Muon, cross_entropy, gradient_clipping, lr_scheduler, decode
from cs336_systems.flash_attention import FlashAttention
import wandb
from contextlib import nullcontext
import warnings
warnings.filterwarnings(
    "ignore", message=".*'repr'.*Field.*|.*'frozen'.*Field.*")


def setup(rank, world_size, device):
    if device.type == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def get_batch_sharded(dataset: np.ndarray | str, batch_size: int, context_length: int, device: str, rank, world_size, step) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(dataset, str):
        dataset = np.load(dataset, mmap_mode='r')
    random = np.random.default_rng(seed=step)
    starting_indices = random.integers(
        0, len(dataset)-context_length, batch_size)
    batch_size_per_rank = batch_size//world_size
    starting_indices = starting_indices[batch_size_per_rank *
                                        rank: batch_size_per_rank * rank + batch_size_per_rank]
    inputs = [dataset[i:i+context_length] for i in starting_indices]
    targets = [dataset[i+1:i+1+context_length] for i in starting_indices]
    inputs = torch.tensor(np.stack(inputs), device=device, dtype=torch.int32)
    targets = torch.tensor(np.stack(targets), device=device, dtype=torch.int32)
    return inputs, targets


def gradient_clipping_sharded(model, max_l2_norm, device):

    local_grad_norm = torch.tensor(0.0, device=device)
    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        local_grad_norm += torch.sum(parameter.grad ** 2)

    dist.all_reduce(local_grad_norm, op=dist.ReduceOp.SUM)

    grad_norm = torch.sqrt(local_grad_norm).item()

    if grad_norm > max_l2_norm:
        scale = max_l2_norm/grad_norm
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            parameter.grad.mul_(scale)


def save_checkpoint_distributed(rank: int, model: torch.nn.Module, optimizers: tuple[torch.optim.Optimizer], iteration: int, out: str, run_id=None, model_config=None) -> None:

    dist.barrier()
    if not isinstance(optimizers, (tuple, list)):
        optimizers = (optimizers,)

    optim_state_dict = {f"optimizer_{i}": opt.state_dict()
                        for i, opt in enumerate(optimizers)}

    torch.save({"model": model.state_dict(),
               **optim_state_dict, "iteration": iteration, "run_id": run_id, "model_config": model_config}, out+f"_rank{rank}.pt")

    dist.barrier()


def load_checkpoint_distributed(
        rank: int,
        src: str,
        model: torch.nn.Module = None,
        optimizers: tuple[torch.optim.Optimizer] = None) -> tuple[int, typing.Any, typing.Optional[dict]] | None:

    if src is None or not os.path.exists(src):
        if rank == 0:
            print("No model to load - starting from scratch")
        return (0, None, None)

    checkpoint = torch.load(src+f"_rank{rank}.pt")

    checkpoint["model"] = {k.replace("_orig_mod.", ""): v for k,
                           v in checkpoint["model"].items()}
    if model is not None:
        model.load_state_dict(checkpoint["model"])

    if optimizers is not None:
        if not isinstance(optimizers, (tuple, list)):
            optimizers = (optimizers,)
        for i, optimizer in enumerate(optimizers):
            optimizer.load_state_dict(checkpoint[f"optimizer_{i}"])

    if rank == 0:

        print(f"Successfully loaded previous model checkpoint from {src}")

    return (checkpoint["iteration"], checkpoint["run_id"], checkpoint["model_config"])


class MultiHeadFlashSelfAttention(nn.Module):

    # Use Flash attention here
    # We can then set import cs336_basics.model  and set cs336_basics.model.MultiHeadSelfAttention = MultiHeadFlashSelfAttention

    def __init__(self, d_model: int, num_heads: int, rope: nn.Module = None, device=None, dtype=None, max_seq_len: int = 512, mask: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.mask = mask
        if rope is not None:
            self.rope = rope
        self.WQ = nn.Parameter(torch.empty(
            d_model, d_model, device=device, dtype=dtype))
        self.WK = nn.Parameter(torch.empty(
            d_model, d_model, device=device, dtype=dtype))
        self.WV = nn.Parameter(torch.empty(
            d_model, d_model, device=device, dtype=dtype))
        self.WO = nn.Parameter(torch.empty(
            d_model, d_model, device=device, dtype=dtype))

        sigma = math.sqrt(2/(d_model + d_model))
        nn.init.trunc_normal_(
            self.WQ, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(
            self.WK, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(
            self.WV, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(
            self.WO, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x: torch.Tensor, token_positions: int = None):

        batch, seq_len, _ = x.shape

        # Apply projection matrix and split d_model into n_heads, d_head + move n_heads to the front as part of the batch dimension
        Q = (x @ self.WQ.T).view(batch, seq_len, self.num_heads,
                                 self.d_head).transpose(-2, -3).reshape(batch * self.num_heads, seq_len, self.d_head)
        K = (x @ self.WK.T).view(batch, seq_len, self.num_heads,
                                 self.d_head).transpose(-2, -3).reshape(batch * self.num_heads, seq_len, self.d_head)
        V = (x @ self.WV.T).view(batch, seq_len, self.num_heads,
                                 self.d_head).transpose(-2, -3).reshape(batch * self.num_heads, seq_len, self.d_head)

        # Apply RoPE
        if hasattr(self, "rope"):
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)

        # Flash attention
        # (batch * heads, seq, d_head)
        output = FlashAttention.apply(Q, K, V, self.mask)

        # Reshape back
        output = output.view(batch, self.num_heads, seq_len, self.d_head)
        output = output.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return output @ self.WO.transpose(-2, -1)


model_module.MultiHeadSelfAttention = MultiHeadFlashSelfAttention


def val_step_distributed(model, val_path, batch_size, context_length, device, rank, world_size, step, norm, torch_amp_autocast):

    with torch.inference_mode():
        losses = []

        for _ in range(5):

            inputs, targets = get_batch_sharded(
                dataset=val_path, batch_size=batch_size, context_length=context_length, device=device, rank=rank, world_size=world_size, step=step)

            model.eval()

            with torch_amp_autocast():
                logits = model(inputs, norm)
                loss = cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )

            losses.append(loss)

        local_loss = torch.mean(torch.stack(losses))

        dist.all_reduce(local_loss, op=dist.ReduceOp.AVG)

        return local_loss


def train_step_distributed(model, optimizers, train_path, batch_size, context_length, device, rank, world_size, step, norm, max_l2_norm,  shard_gradient, torch_amp_autocast):

    inputs, targets = get_batch_sharded(
        dataset=train_path, batch_size=batch_size, context_length=context_length, device=device, rank=rank, world_size=world_size, step=step)

    model.train()
    with torch_amp_autocast():
        logits = model(inputs, norm)
        loss = cross_entropy(logits.view(-1, logits.size(-1)),
                             targets.view(-1))

    for optimizer in optimizers:
        optimizer.zero_grad()

    loss.backward()

    model.finish_gradient_synchronization()

    if shard_gradient:
        gradient_clipping_sharded(model, max_l2_norm, device)

    else:
        gradient_clipping(model.parameters(), max_l2_norm)

    for optimizer in optimizers:
        optimizer.step()

    if rank == 0:
        return loss.item()


def train_distributed(
        rank,
        world_size,
        shard_optimizer=False,
        shard_gradient=False,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 16,
        d_ff: int = 1344,
        rope_theta: float = 10000.0,
        vocab_size: int = 10000,
        train_path: str = "../data/TinyStoriesV2-GPT4-train.npy",
        val_path: str = "../data/TinyStoriesV2-GPT4-val.npy",
        tokenizer_path: str = "../checkpoints/tokenizer_tiny_stories.model",
        batch_size: int = 32,
        context_length: int = 256,
        steps: int = 5000,
        max_learning_rate: float = 5e-4,
        max_l2_norm: float = 1.0,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
        load_model_path: str | None = None,
        save_model_path: str | None = None,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        norm=True,
        rope=True,
        cautious_weight_decay=False,
        use_muon=False,
        val_interval: int = 250,
        save_interval: int = 1000,
        mixed_precision_dtype=torch.float16):

    setup(rank, world_size, device)

    # Seed for reproducible model initialization across all ranks
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    if device.type == "cuda":
        torch.cuda.set_device(rank)
        data_device = "cuda"
    else:
        data_device = device.type

    if mixed_precision_dtype is None:
        torch_amp_autocast = nullcontext
    else:
        def torch_amp_autocast():
            return torch.amp.autocast("cuda", dtype=mixed_precision_dtype)

    cosine_cycle_iters = steps
    warmup_iters = min(100, int(0.02*steps))
    min_learning_rate = 0.1 * max_learning_rate

    model = Transformer(vocab_size, context_length, d_model, num_layers,
                        num_heads, d_ff, rope_theta, None, data_device, dtype, norm=norm, rope=rope)

    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    iteration, run_id, model_config = load_checkpoint_distributed(
        rank, load_model_path, model, None)

    if model_config is None:
        model_config = {"vocab_size": vocab_size, "context_length": context_length, "d_model": d_model, "num_layers": num_layers,
                        "num_heads": num_heads, "d_ff": d_ff, "rope_theta": rope_theta, "weights": None, "device": data_device, "dtype": dtype}

    model = DDPParameter(
        model, sharded=shard_gradient, use_muon=use_muon)

    if use_muon:
        parameters_adam = []
        parameters_muon = []

        for name, param in model.named_parameters():
            if param.ndim >= 2 and name not in ("module.output.W", "module.embedding.E"):
                parameters_muon.append(param)
            else:
                parameters_adam.append(param)

        optimizer_adam_ddp = ShardOptimizer(
            parameters_adam, AdamW, max_learning_rate, weight_decay, betas, cautious_weight_decay)
        optimizer_muon_ddp = ShardOptimizer(
            parameters_muon, Muon, max_learning_rate,  weight_decay, betas[1], cautious_weight_decay)

        optimizers = (optimizer_adam_ddp, optimizer_muon_ddp)

    else:
        optimizer_adam_ddp = ShardOptimizer(
            model.parameters(), AdamW, max_learning_rate, weight_decay, betas, cautious_weight_decay)

        optimizers = (optimizer_adam_ddp, )

    load_checkpoint_distributed(rank, load_model_path, model, optimizers)

    if rank == 0:
        print("Compiling model")
    if data_device == "cuda":
        model = torch.compile(model)

    if rank == 0:
        print("Starting training")
        num_parameters = model.module.get_parameters(verbose=True)
        model.module.get_training_memory(
            verbose=True, batch_size=batch_size, use_muon=use_muon, ddp=True, world_size=world_size, shard_gradient=shard_gradient, shard_optimizer=shard_optimizer, mixed_precision=mixed_precision_dtype is not None)
        model.module.get_training_time(verbose=True, steps=steps, ddp=True,
                                       batch_size=batch_size, world_size=world_size)
        print(f"Training on {data_device}")

        wandb.init(
            project="Distributed Transformer",
            name="small-model",
            id=run_id,
            resume="allow",
            config={
                "max_lr": max_learning_rate,
                "batch_size": batch_size,
                "model parameters in Million": round(num_parameters/1e6, 2),
                "steps": steps,
                "total_tokens": steps*batch_size*context_length,
                "optimizer": "muon" if use_muon else "adam",
                "ddp": True,
                "zero1": shard_optimizer,
                "zero2": shard_gradient,
                "world_size": world_size
            }
        )

        run_id = wandb.run.id
    else:
        run_id = 0

    best_val_loss = float("inf")

    for step in range(iteration, steps):

        lr = lr_scheduler(step, max_learning_rate,
                          min_learning_rate, warmup_iters, cosine_cycle_iters)

        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        loss = train_step_distributed(model, optimizers, train_path, batch_size, context_length,
                                      device, rank, world_size, step, norm, max_l2_norm, shard_gradient, torch_amp_autocast)

        if step % 10 == 0:
            param_norm = torch.sqrt(
                sum(p.norm()**2 for p in model.parameters())).item()

            grad_norm = torch.sqrt(
                sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)).item()

            if rank == 0:
                wandb.log({
                    "train/loss": loss,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "weight_norm": param_norm

                }, step=step)

        if step % val_interval == 0:
            val_loss = val_step_distributed(
                model, val_path, batch_size, context_length, device, rank, world_size, step, norm, torch_amp_autocast)

            if rank == 0:
                wandb.log({
                    "val/loss": val_loss,
                    "val/perplexity": math.exp(val_loss),
                }, step=step)

            # Only compare against best_val_loss at save_interval steps
            if step % save_interval == 0 and save_model_path is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint_distributed(
                    rank, model, optimizers, step, save_model_path, run_id, model_config)

                if rank == 0:
                    print(
                        "Sampling a model generation to see current performance - Once upon a time: ...")
                    # Use uncompiled model for inference (avoids recompilation overhead)
                    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                    decode(base_model, tokenizer=tokenizer,
                           x="Once upon a time", num_tokens=min(256, context_length - 20),
                           temperature=0.8, top_p_threshold=0.9, norm=norm, device=data_device)

    if rank == 0:
        wandb.finish()

    dist.destroy_process_group()


def run_distributed_training(
        # Hardware
        world_size: int = 2,
        device: torch.device = None,
        # Model architecture
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: int = 3072,
        rope_theta: float = 10000.0,
        vocab_size: int = 10000,
        context_length: int = 256,
        # Training
        steps: int = 5000,
        batch_size: int = 32,
        max_learning_rate: float = 5e-4,
        max_l2_norm: float = 1.0,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        # Distributed settings
        shard_optimizer: bool = True,
        shard_gradient: bool = True,
        # Optimizer
        use_muon: bool = False,
        cautious_weight_decay: bool = False,
        # Normalization and RoPE
        norm: bool = True,
        rope: bool = True,
        # Precision
        dtype: torch.dtype = torch.float32,
        mixed_precision_dtype: torch.dtype = torch.float16,
        # Paths
        train_path: str = "../data/TinyStoriesV2-GPT4-train.npy",
        val_path: str = "../data/TinyStoriesV2-GPT4-val.npy",
        tokenizer_path: str = "../checkpoints/tokenizer_tiny_stories.model",
        load_model_path: str = None,
        save_model_path: str = None,
        # Logging intervals
        val_interval: int = 250,
        save_interval: int = 1000,
):
    """
    Run distributed training with the given configuration.
    """
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print("Running on GPU(s)")

    mp.spawn(
        fn=train_distributed,
        args=(
            world_size,
            shard_optimizer,
            shard_gradient,
            d_model,
            num_layers,
            num_heads,
            d_ff,
            rope_theta,
            vocab_size,
            train_path,
            val_path,
            tokenizer_path,
            batch_size,
            context_length,
            steps,
            max_learning_rate,
            max_l2_norm,
            device,
            dtype,
            load_model_path,
            save_model_path,
            weight_decay,
            betas,
            norm,
            rope,
            cautious_weight_decay,
            use_muon,
            val_interval,
            save_interval,
            mixed_precision_dtype,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    run_distributed_training(
        world_size=2,
        d_model=1280,
        num_layers=24,
        num_heads=10,
        d_ff=3456,
        vocab_size=32000,
        context_length=1024,
        rope_theta=10000,
        steps=10,
        batch_size=16,
        shard_optimizer=True,
        shard_gradient=True,
        use_muon=True,
        cautious_weight_decay=True,
        tokenizer_path="checkpoints/tokenizer_owt.model",
        train_path="../data/owt_train.npy",
        val_path="../data/owt_valid.npy",
    )
