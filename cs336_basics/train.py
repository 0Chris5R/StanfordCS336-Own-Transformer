"""Training utilities for the CS336 basics assignment."""

import torch
import torch.nn.functional as F
import math
import json
from typing import Iterable
import numpy as np
import os
import typing
from cs336_basics.model import Transformer, softmax
import wandb
from cs336_basics.tokenizer import BPETokenizer


def cross_entropy(x: torch.Tensor, targets: torch.Tensor) -> float:

    # Using Log Sum exp trick:

    dtype = x.dtype
    x = x.to(torch.float32)

    m = torch.max(x, dim=-1, keepdim=True).values
    return (torch.mean((-x + m + torch.log(torch.sum(torch.exp(x-m), dim=-1, keepdim=True)))[torch.arange(x.shape[0]), targets])).to(dtype)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-7, cautious_weight_decay=False):

        defaults = {"lr": lr, "beta1": betas[0],
                    "beta2": betas[1], "eps": eps, "lam": weight_decay}
        super().__init__(params, defaults)
        self.cautious_weight_decay = cautious_weight_decay

    @torch.no_grad()
    def step(self, closure=None):

        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            lam = group["lam"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state["velocity"] = torch.zeros_like(p)
                    state["momentum"] = torch.zeros_like(p)
                    state["step"] = 1

                v = state["velocity"]
                m = state["momentum"]
                step = state["step"]

                grad = p.grad.data

                m.lerp_(grad, 1-beta1)
                v.lerp_(grad.square(), 1-beta2)

                bias_1 = 1-beta1**step
                bias_2 = 1-beta2**step

                step_size = lr/bias_1

                denom = torch.sqrt(v/bias_2) + eps

                update = m/denom

                # cautious weight decay: only applies weight decay for parameters where update direction and weights align
                if self.cautious_weight_decay is True:

                    p.add_(p * ((p * update) >= 0).float(), alpha=-lr*lam)

                # normal decoupled weigth decay:
                else:
                    p.mul_(1 - lr * lam)

                p.add_(update, alpha=-step_size)

                state["step"] += 1

        return loss


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, mu=0.95,  cautious_weight_decay=False):
        self.cautious_weight_decay = cautious_weight_decay
        defaults = {"lr": lr, "lam": weight_decay, "mu": mu}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            lam = group["lam"]
            mu = group["mu"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                    state["step"] = 1

                m = state["momentum"]

                grad = p.grad.data

                m.mul_(mu)
                m.add_(grad)

                # Nesterov with lookahead:
                m_temp = m * mu + grad
                update = (self._newton_schulz(m_temp))

                # scale the RMS of Muon to the same range of ADAMW (empirically 0.2 to 0.4) this way we can use the lr and weight_decay from ADAMW
                step_size = 0.2 * \
                    math.sqrt(max(m.size(0), m.size(1))) * lr

                # cautious weight decay: only applies weight decay for parameters where update direction and weights align
                if self.cautious_weight_decay is True:

                    p.add_(p * ((p * update) >= 0).float(), alpha=-lr*lam)

                # normal decoupled weigth decay:
                else:
                    p.mul_(1 - lr * lam)

                p.add_(update, alpha=-step_size)

        return loss

    @torch.no_grad()
    def _newton_schulz(self, g: torch.tensor, steps=5, eps=1e-7):
        x = g / (torch.linalg.norm(g, 'fro')+eps)
        a_cof, b_cof, c_cof = 3.4445, -4.7750, 2.0315
        if g.size(0) > g.size(1):
            x = x.T
        # newton Schulz Formula: a*X + b*(X@X.T)@X + c*(X@X.T)**2 @ X
        for _ in range(steps):
            a = x @ x.T
            b = b_cof * a + c_cof * a @ a
            x = a_cof * x + b @ x

        if g.size(0) > g.size(1):
            x = x.T

        return x


def lr_scheduler(t, max_learning_rate: float,
                 min_learning_rate: float,
                 warmup_iters: int,
                 cosine_cycle_iters: int,) -> float:

    if t < warmup_iters:
        return (t/warmup_iters)*max_learning_rate

    if t > cosine_cycle_iters:
        return min_learning_rate

    return min_learning_rate + 0.5 * (1 + math.cos((t - warmup_iters)/(cosine_cycle_iters-warmup_iters) * math.pi)) * (max_learning_rate-min_learning_rate)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:

    grad_norm = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue

        grad_norm += torch.sum(parameter.grad ** 2)

    grad_norm = torch.sqrt(grad_norm).item()

    if grad_norm > max_l2_norm:
        scale = max_l2_norm/grad_norm
        for parameter in parameters:
            if parameter.grad is None:
                continue
            parameter.grad.mul_(scale)


def get_batch(dataset: np.ndarray | str, batch_size: int, context_length: int, device: str
              ) -> tuple[torch.Tensor, torch.Tensor]:

    if isinstance(dataset, str):
        dataset = np.load(dataset, mmap_mode='r')

    random = np.random.default_rng()
    starting_indices = random.integers(
        0, len(dataset)-context_length, batch_size)

    inputs = [dataset[i:i+context_length] for i in starting_indices]
    targets = [dataset[i+1:i+1+context_length] for i in starting_indices]

    inputs = torch.tensor(np.stack(inputs), device=device, dtype=torch.int32)
    targets = torch.tensor(np.stack(targets), device=device, dtype=torch.int32)

    return inputs, targets


def save_checkpoint(model: torch.nn.Module,
                    optimizers: tuple[torch.optim.Optimizer],
                    iteration: int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], run_id=None, model_config=None) -> None:

    if not isinstance(optimizers, (tuple, list)):
        optimizers = (optimizers,)

    optim_state_dict = {f"optimizer_{i}": opt.state_dict()
                        for i, opt in enumerate(optimizers)}

    torch.save({"model": model.state_dict(),
               **optim_state_dict, "iteration": iteration, "run_id": run_id, "model_config": model_config}, out)


def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module = None,
        optimizers: tuple[torch.optim.Optimizer] = None) -> tuple[int, typing.Any, typing.Optional[dict]]:

    if src is None or not os.path.exists(src):
        print("No model to load - starting from scratch")
        return (0, None, None)

    checkpoint = torch.load(src)
    checkpoint["model"] = {k.replace("_orig_mod.", ""): v for k,
                           v in checkpoint["model"].items()}
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    if optimizers is not None:
        if not isinstance(optimizers, (tuple, list)):
            optimizers = (optimizers,)
        for i, optimizer in enumerate(optimizers):
            optimizer.load_state_dict(checkpoint[f"optimizer_{i}"])

    print(f"Successfully loaded previous model checkpoint from {src}")

    return (checkpoint["iteration"], checkpoint["run_id"], checkpoint["model_config"])


def train_together(
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
    device: torch.device = torch.device("mps"),
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

) -> None:

    cosine_cycle_iters = steps
    warmup_iters = min(100, int(0.02*steps))
    min_learning_rate = 0.1 * max_learning_rate

    model = Transformer(vocab_size, context_length, d_model, num_layers,
                        num_heads, d_ff, rope_theta, None, device, dtype, norm=norm, rope=rope)

    if use_muon:
        muon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Apply Muon only to weight matrices (2D tensors) in Linear layers
            if len(param.shape) == 2 and name not in ("output.W", "embedding.E"):
                muon_params.append(param)
            else:
                adamw_params.append(param)
        muon = Muon(muon_params, max_learning_rate, weight_decay,
                    betas[1], cautious_weight_decay)
        adamw = AdamW(adamw_params, max_learning_rate, weight_decay,
                      betas, eps=1e-7, cautious_weight_decay=cautious_weight_decay)
        optimizers = (muon, adamw)
    else:
        optimizer = AdamW(model.parameters(), max_learning_rate,
                          weight_decay, betas, eps=1e-7, cautious_weight_decay=cautious_weight_decay)
        optimizers = (optimizer,)

    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    iteration, run_id, model_config = load_checkpoint(
        load_model_path, model, optimizers)

    if model_config is None:
        model_config = {"vocab_size": vocab_size, "context_length": context_length, "d_model": d_model, "num_layers": num_layers,
                        "num_heads": num_heads, "d_ff": d_ff, "rope_theta": rope_theta, "weights": None, "device": device, "dtype": dtype}

    print("Starting training")
    num_parameters = model.get_parameters(verbose=True)
    model.get_training_memory(
        verbose=True, batch_size=batch_size, use_muon=use_muon)
    model.get_training_time(verbose=True, steps=steps, batch_size=batch_size)
    print(f"Training on {model.device}")

    print("Compiling model")
    if device.type == "cuda":
        model = torch.compile(model)

    wandb.init(
        project="Transformer-from-scratch",
        name="small-model",
        id=run_id,
        resume="allow",
        config={
            "max_lr": max_learning_rate,
            "batch_size": batch_size,
            "model parameters in Million": round(num_parameters/1e6, 2),
            "steps": steps,
            "total_tokens": steps*batch_size*context_length,
            "optimizer": "muon" if use_muon else "adam"
        }
    )

    run_id = wandb.run.id

    best_val_loss = float("inf")

    for step in range(iteration, steps):

        lr = lr_scheduler(step, max_learning_rate,
                          min_learning_rate, warmup_iters, cosine_cycle_iters)

        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        loss = train_step(model, optimizers, train_path,
                          batch_size, context_length, device, max_l2_norm, norm=norm)

        if step % 10 == 0:
            param_norm = torch.sqrt(
                sum(p.norm()**2 for p in model.parameters())).item()

            grad_norm = torch.sqrt(
                sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)).item()

            wandb.log({
                "train/loss": loss,
                "lr": lr,
                "grad_norm": grad_norm,
                "weight_norm": param_norm

            }, step=step)

        if step % val_interval == 0:
            val_loss = val_step(model, val_path, batch_size,
                                context_length, device, norm=norm)

            wandb.log({
                "val/loss": val_loss,
                "val/perplexity": math.exp(val_loss),
            }, step=step)

            # Only compare against best_val_loss at save_interval steps
            if step % save_interval == 0 and save_model_path is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizers, step,
                                save_model_path, run_id, model_config)

                print(
                    "Sampling a model generation to see current performance - Once upon a time: ...")
                decode(model, tokenizer=tokenizer,
                       x="Once upon a time", num_tokens=20, temperature=0.8, top_p_threshold=0.9, norm=norm)

    wandb.finish()


def train_step(model, optimizers: tuple, train_path, batch_size, context_length, device, max_l2_norm, norm=True) -> float:

    inputs, targets = get_batch(
        dataset=train_path, batch_size=batch_size, context_length=context_length, device=device)

    model.train()
    logits = model(inputs, norm)
    loss = cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1))

    for optimizer in optimizers:
        optimizer.zero_grad()

    loss.backward()

    gradient_clipping(model.parameters(), max_l2_norm)

    for optimizer in optimizers:
        optimizer.step()

    return loss.item()


def val_step(model, val_path, batch_size, context_length, device, norm=True) -> float:

    with torch.inference_mode():

        losses = []

        for _ in range(5):

            inputs, targets = get_batch(
                dataset=val_path, batch_size=batch_size, context_length=context_length, device=device)

            model.eval()

            logits = model(inputs, norm)
            loss = cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            losses.append(loss)

        return torch.mean(torch.stack(losses)).item()


def decode(model_path, tokenizer, x, num_tokens, temperature=1, top_p_threshold=1, norm=True, rope=True, device=None) -> None:

    # Can take in either model path or model

    if isinstance(model_path, str):

        _, _, config = load_checkpoint(model_path, None, None)
        # Instantiate the Transformer using the saved config
        model = Transformer(
            vocab_size=config["vocab_size"],
            context_length=config["context_length"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=config["rope_theta"],
            weights=config["weights"],
            device=config["device"],
            dtype=config["dtype"],
            norm=norm,
            rope=rope

        )

        _, _, _ = load_checkpoint(model_path, model, None)
        device = config["device"]

    else:
        model = model_path
        if device is None:
            device = next(model.parameters()).device

    print(x, end="", flush=True)

    # our model takes tensor(batch, seq) as input
    x = torch.tensor(tokenizer.encode(x), device=device).unsqueeze(0)

    with torch.inference_mode():

        model.eval()

        for _ in range(num_tokens):

            logits = model(x, norm)[0, -1, :]

            probabilities = softmax(logits, d=-1, temperature=temperature)

            # nucleus sampling:
            probabilities, indices = torch.sort(probabilities, descending=True)
            cumulative_probs = torch.cumsum(probabilities, dim=0)
            mask = cumulative_probs <= top_p_threshold
            # if the model is so confident that the first probability already exceeds the threshold we need to include at least that token
            if not mask.any():
                mask[0] = True
            top_p = probabilities[mask]
            top_p = top_p / top_p.sum()

            # sample:
            token_id = indices[torch.multinomial(top_p, num_samples=1)]

            # stop if token is <|endoftext|>
            if token_id.item() == 256:
                return

            token = tokenizer.decode([token_id.item()])

            print(token, end="", flush=True)

            x = torch.cat([x, token_id.unsqueeze(0)], dim=1)

        print()
