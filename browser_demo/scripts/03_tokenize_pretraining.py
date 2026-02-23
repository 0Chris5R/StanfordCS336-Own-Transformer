#!/usr/bin/env python3
"""Tokenize pretraining data using the 16K tokenizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.tokenizer import BPETokenizer

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"


def main():
    tokenizer_path = CHECKPOINT_DIR / "tokenizer_16k.model"
    input_path = DATA_DIR / "fineweb-edu" / "pretraining_raw.txt"
    output_path = DATA_DIR / "fineweb-edu" / "pretraining.npy"

    # Check prerequisites
    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        print("Run: uv run python scripts/01_train_tokenizer.py first")
        return

    if not input_path.exists():
        print(f"ERROR: Pretraining data not found at {input_path}")
        print("Run: uv run python scripts/02_download_pretraining.py first")
        return

    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = BPETokenizer(special_tokens=[])
    tokenizer.load(str(tokenizer_path))
    print(f"  Vocab size: {len(tokenizer.vocab)}")

    # Tokenize
    print(f"\nTokenizing {input_path}...")
    print("  This may take 10-30 minutes depending on data size...")
    tokenizer.tokenize_and_save(str(input_path), str(output_path))

    # Report stats
    import numpy as np
    tokens = np.load(output_path)
    num_tokens = len(tokens)

    print(f"\nTokenization complete!")
    print(f"  Total tokens: {num_tokens:,}")
    print(f"  Output file: {output_path}")
    print(f"  Output size: {output_path.stat().st_size / (1024**3):.2f} GB")

    # Calculate training stats
    batch_size = 64
    context_length = 256
    tokens_per_step = batch_size * context_length
    total_steps = num_tokens // tokens_per_step

    print(f"\nTraining estimates (batch={batch_size}, ctx={context_length}):")
    print(f"  Tokens per step: {tokens_per_step:,}")
    print(f"  Total steps: {total_steps:,}")

    # Split into train/val
    val_tokens = 1_000_000  # 1M tokens for validation
    train_tokens = num_tokens - val_tokens

    train_path = DATA_DIR / "fineweb-edu" / "pretraining_train.npy"
    val_path = DATA_DIR / "fineweb-edu" / "pretraining_val.npy"

    print(f"\nSplitting into train/val...")
    np.save(train_path, tokens[:train_tokens])
    np.save(val_path, tokens[train_tokens:])

    print(f"  Train: {train_tokens:,} tokens -> {train_path.name}")
    print(f"  Val: {val_tokens:,} tokens -> {val_path.name}")

    print("\nNext: Run scripts/04_pretrain.py")


if __name__ == "__main__":
    main()
