import torch
import gc
import time
from cs336_basics.train import train_together
from cs336_systems.train_distributed import run_distributed_training


CONFIG = {
    "d_model": 1024,
    "num_layers": 10,
    "num_heads": 8,
    "d_ff": 2688,
    "vocab_size": 32000,
    "context_length": 512,
    "rope_theta": 10000,
    "steps": 500,
    "batch_size": 12,
    "max_learning_rate": 1e-3,
    "weight_decay": 0.01,
    "betas": (0.9, 0.95),
    "use_muon": True,
    "cautious_weight_decay": True,
    "train_path": "/kaggle/input/datasets/cr0512/owt-npy/owt_train.npy",
    "val_path": "/kaggle/input/datasets/cr0512/owt-npy/owt_valid.npy",
    "tokenizer_path": "checkpoints/tokenizer_owt.model",
}


def benchmark_single_process():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    start = time.time()
    train_together(
        d_model=CONFIG["d_model"],
        num_layers=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        d_ff=CONFIG["d_ff"],
        vocab_size=CONFIG["vocab_size"],
        context_length=CONFIG["context_length"],
        rope_theta=CONFIG["rope_theta"],
        steps=CONFIG["steps"],
        batch_size=CONFIG["batch_size"],
        max_learning_rate=CONFIG["max_learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        betas=CONFIG["betas"],
        use_muon=CONFIG["use_muon"],
        cautious_weight_decay=CONFIG["cautious_weight_decay"],
        train_path=CONFIG["train_path"],
        val_path=CONFIG["val_path"],
        tokenizer_path=CONFIG["tokenizer_path"],
        save_model_path="checkpoints/benchmark_single.pt",
        device=torch.device("cuda"),
        dtype=torch.float32,
        val_interval=CONFIG["steps"] + 1,
        save_interval=CONFIG["steps"],
    )
    elapsed = time.time() - start

    gc.collect()
    torch.cuda.empty_cache()

    return elapsed


def benchmark_distributed_world1():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    start = time.time()
    run_distributed_training(
        world_size=1,
        d_model=CONFIG["d_model"],
        num_layers=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        d_ff=CONFIG["d_ff"],
        vocab_size=CONFIG["vocab_size"],
        context_length=CONFIG["context_length"],
        rope_theta=CONFIG["rope_theta"],
        steps=CONFIG["steps"],
        batch_size=CONFIG["batch_size"],
        max_learning_rate=CONFIG["max_learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        betas=CONFIG["betas"],
        use_muon=CONFIG["use_muon"],
        cautious_weight_decay=CONFIG["cautious_weight_decay"],
        train_path=CONFIG["train_path"],
        val_path=CONFIG["val_path"],
        tokenizer_path=CONFIG["tokenizer_path"],
        save_model_path="checkpoints/benchmark_dist_w1",
        shard_optimizer=False,
        shard_gradient=False,
        mixed_precision_dtype=None,
        val_interval=CONFIG["steps"] + 1,
        save_interval=CONFIG["steps"],
    )
    elapsed = time.time() - start

    gc.collect()
    torch.cuda.empty_cache()

    return elapsed


def benchmark_distributed_world2():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    start = time.time()
    run_distributed_training(
        world_size=2,
        d_model=CONFIG["d_model"],
        num_layers=CONFIG["num_layers"],
        num_heads=CONFIG["num_heads"],
        d_ff=CONFIG["d_ff"],
        vocab_size=CONFIG["vocab_size"],
        context_length=CONFIG["context_length"],
        rope_theta=CONFIG["rope_theta"],
        steps=CONFIG["steps"],
        batch_size=CONFIG["batch_size"],
        max_learning_rate=CONFIG["max_learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        betas=CONFIG["betas"],
        use_muon=CONFIG["use_muon"],
        cautious_weight_decay=CONFIG["cautious_weight_decay"],
        train_path=CONFIG["train_path"],
        val_path=CONFIG["val_path"],
        tokenizer_path=CONFIG["tokenizer_path"],
        save_model_path="checkpoints/benchmark_dist_w2",
        shard_optimizer=False,
        shard_gradient=False,
        mixed_precision_dtype=None,
        val_interval=CONFIG["steps"] + 1,
        save_interval=CONFIG["steps"],
    )
    elapsed = time.time() - start

    gc.collect()
    torch.cuda.empty_cache()

    return elapsed


def compare_weights():
    single_weights = torch.load("checkpoints/benchmark_single.pt")["model"]
    dist_w1_weights = torch.load(
        "checkpoints/benchmark_dist_w1_rank0.pt")["model"]
    dist_w2_weights = torch.load(
        "checkpoints/benchmark_dist_w2_rank0.pt")["model"]

    single_weights = {k.replace("_orig_mod.", "")
                                : v for k, v in single_weights.items()}
    dist_w1_weights = {k.replace("_orig_mod.", "").replace(
        "module.", ""): v for k, v in dist_w1_weights.items()}
    dist_w2_weights = {k.replace("_orig_mod.", "").replace(
        "module.", ""): v for k, v in dist_w2_weights.items()}

    print("\n" + "="*60)
    print("WEIGHT COMPARISON")
    print("="*60)

    try:
        for key in single_weights:
            torch.testing.assert_close(
                single_weights[key], dist_w1_weights[key], atol=1e-4, rtol=1e-4)
        print("Single process vs Distributed (world_size=1): MATCH")
    except AssertionError as e:
        print(f"Single process vs Distributed (world_size=1): MISMATCH - {e}")

    try:
        for key in single_weights:
            torch.testing.assert_close(
                single_weights[key], dist_w2_weights[key], atol=1e-4, rtol=1e-4)
        print("Single process vs Distributed (world_size=2): MATCH")
    except AssertionError as e:
        print(f"Single process vs Distributed (world_size=2): MISMATCH - {e}")

    try:
        for key in dist_w1_weights:
            torch.testing.assert_close(
                dist_w1_weights[key], dist_w2_weights[key], atol=1e-4, rtol=1e-4)
        print("Distributed (world_size=1) vs Distributed (world_size=2): MATCH")
    except AssertionError as e:
        print(
            f"Distributed (world_size=1) vs Distributed (world_size=2): MISMATCH - {e}")


def main():
    print("="*60)
    print("BENCHMARK: Single Process vs Distributed Training")
    print("="*60)
    print(
        f"Config: {CONFIG['steps']} steps, batch_size={CONFIG['batch_size']}")
    print(
        f"Model: d={CONFIG['d_model']}, layers={CONFIG['num_layers']}, heads={CONFIG['num_heads']}")
    print("="*60)

    print("\n[1/3] Running single process training...")
    time_single = benchmark_single_process()
    print(
        f"Single process: {time_single:.2f}s ({CONFIG['steps']/time_single:.2f} steps/s)")

    print("\n[2/3] Running distributed training (world_size=1)...")
    time_dist_w1 = benchmark_distributed_world1()
    print(
        f"Distributed w1: {time_dist_w1:.2f}s ({CONFIG['steps']/time_dist_w1:.2f} steps/s)")

    print("\n[3/3] Running distributed training (world_size=2)...")
    time_dist_w2 = benchmark_distributed_world2()
    print(
        f"Distributed w2: {time_dist_w2:.2f}s ({CONFIG['steps']/time_dist_w2:.2f} steps/s)")

    print("\n" + "="*60)
    print("TIMING SUMMARY")
    print("="*60)
    print(f"Single process:           {time_single:.2f}s")
    print(
        f"Distributed (world_size=1): {time_dist_w1:.2f}s (overhead: {(time_dist_w1/time_single - 1)*100:.1f}%)")
    print(
        f"Distributed (world_size=2): {time_dist_w2:.2f}s (speedup: {time_single/time_dist_w2:.2f}x)")

    compare_weights()


if __name__ == "__main__":
    main()
