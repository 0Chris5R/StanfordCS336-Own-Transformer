#!/usr/bin/env python3
"""Pretrain the 70M model on 2B tokens of FineWeb-Edu.

This is the critical foundation phase. Expected time:
- ~1-2 days on single A100
- ~3-5 days on Apple M-series (MPS)

The model learns grammar, vocabulary, and coherent generation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.train import train_together

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "fineweb-edu"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# Model config for ~70M parameters
MODEL_CONFIG = {
    "d_model": 640,
    "num_layers": 6,
    "num_heads": 10,
    "d_ff": 1728,
    "rope_theta": 10000.0,
    "vocab_size": 16000,
    "context_length": 256,
}


def main():
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Pretrain 70M model")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of steps (default: auto from data)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=6e-4,
                        help="Max learning rate (default: 6e-4)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")

    args = parser.parse_args()

    # Paths
    train_path = DATA_DIR / "pretraining_train.npy"
    val_path = DATA_DIR / "pretraining_val.npy"
    tokenizer_path = CHECKPOINT_DIR / "tokenizer_16k.model"
    save_path = CHECKPOINT_DIR / "pretrain.pt"

    # Check prerequisites
    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        print("Run scripts/03_tokenize_pretraining.py first")
        return

    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        print("Run scripts/01_train_tokenizer.py first")
        return

    # Calculate steps from data if not specified
    if args.steps is None:
        tokens = np.load(train_path, mmap_mode='r')
        num_tokens = len(tokens)
        tokens_per_step = args.batch_size * MODEL_CONFIG["context_length"]
        args.steps = num_tokens // tokens_per_step
        print(f"Auto-calculated steps: {args.steps:,} ({num_tokens:,} tokens)")

    print("\n" + "=" * 60)
    print("PHASE 1: PRETRAINING")
    print("=" * 60)
    print(f"Model: {sum(p for p in MODEL_CONFIG.values() if isinstance(p, int))}...")
    print(f"Steps: {args.steps:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Resume: {args.resume}")
    print("=" * 60)

    # Determine load path for resume
    load_path = str(save_path) if args.resume and save_path.exists() else None

    train_together(
        **MODEL_CONFIG,
        train_path=str(train_path),
        val_path=str(val_path),
        tokenizer_path=str(tokenizer_path),
        batch_size=args.batch_size,
        steps=args.steps,
        max_learning_rate=args.lr,
        weight_decay=0.1,
        max_l2_norm=1.0,
        load_model_path=load_path,
        save_model_path=str(save_path),
        resume=args.resume,
        val_interval=1000,
        save_interval=5000,
        use_muon=True,
        cautious_weight_decay=True,
    )

    print("\n" + "=" * 60)
    print("PRETRAINING COMPLETE")
    print(f"Model saved to: {save_path}")
    print("=" * 60)
    print("\nNext: Run scripts/05_format_sft.py")


if __name__ == "__main__":
    main()
