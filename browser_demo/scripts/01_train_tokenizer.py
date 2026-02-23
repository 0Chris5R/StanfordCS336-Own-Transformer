#!/usr/bin/env python3
"""Train a 16K vocabulary tokenizer with special tokens for chat format.

The tokenizer includes:
- 256 byte tokens (base)
- 8 special tokens for chat format
- ~15,736 BPE merges

Training data: FineWeb-Edu + Chris Q&A/Refusals (repeated for domain word coverage)
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.tokenizer import BPETokenizer

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# Special tokens for chat format
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|think|>",
    "<|answer|>",
    "<|refuse|>",
    "<|pad|>",
]


def create_training_data(fineweb_path: Path, output_path: Path, chris_repetitions: int = 20):
    """Combine FineWeb + Chris data for tokenizer training.

    Uses file copy + append to avoid any string manipulation issues.
    """
    import shutil

    print("Creating tokenizer training data...")

    # First, copy the original FineWeb file (preserves exact encoding/format)
    print(f"  Copying FineWeb to {output_path}...")
    shutil.copy(fineweb_path, output_path)

    # Load Chris Q&A
    qa_path = DATA_DIR / "generated" / "chris_qa.json"
    refusal_path = DATA_DIR / "generated" / "chris_refusals.json"

    chris_lines = []
    if qa_path.exists():
        with open(qa_path, encoding='utf-8') as f:
            qa_data = json.load(f)
        for item in qa_data:
            chris_lines.append(f"{item['question']} {item['answer']}")
        print(f"  Loaded {len(qa_data)} Q&A pairs")

    if refusal_path.exists():
        with open(refusal_path, encoding='utf-8') as f:
            refusal_data = json.load(f)
        for item in refusal_data:
            chris_lines.append(f"{item['question']} {item['answer']}")
        print(f"  Loaded {len(refusal_data)} refusal pairs")

    # Append Chris data to the copied file
    print(f"  Appending Chris data (x{chris_repetitions})...")
    with open(output_path, 'a', encoding='utf-8') as f:
        for _ in range(chris_repetitions):
            for line in chris_lines:
                f.write(line + '\n')

    final_size = output_path.stat().st_size
    print(f"  Total size: {final_size / 1024 / 1024:.1f} MB")

    return output_path


def train_tokenizer(input_path: Path, vocab_size: int = 16000):
    """Train BPE tokenizer, then add special tokens."""
    num_merges = vocab_size - 256 - len(SPECIAL_TOKENS)

    print(f"\nTraining tokenizer...")
    print(f"  Target vocab size: {vocab_size}")
    print(f"  Base byte tokens: 256")
    print(f"  BPE merges to learn: {num_merges}")
    print(f"  Special tokens (added after): {len(SPECIAL_TOKENS)}")

    # Train with empty special tokens
    tokenizer = BPETokenizer(special_tokens=[])
    tokenizer.train_tokenizer(str(input_path), vocab_size=256 + num_merges)

    # Add special tokens to the vocabulary
    print(f"\nAdding special tokens...")
    next_id = len(tokenizer.vocab)
    for special in SPECIAL_TOKENS:
        tokenizer.vocab[next_id] = special.encode("utf-8")
        tokenizer.special_tokens.append(special)
        print(f"  {next_id}: {special}")
        next_id += 1

    # Rebuild reverse vocab
    tokenizer._reverse_vocab = tokenizer._build_reverse_vocab()

    return tokenizer


def verify_tokenizer(tokenizer: BPETokenizer):
    """Verify special tokens and check some domain words."""
    print("\nVerifying special tokens...")
    all_good = True
    for special in SPECIAL_TOKENS:
        tokens = tokenizer.encode(special)
        if len(tokens) != 1:
            print(f"  ERROR: '{special}' -> {len(tokens)} tokens (expected 1)")
            all_good = False
        else:
            print(f"  OK: '{special}' -> token {tokens[0]}")

    # Check some domain words (informational only)
    print("\nDomain word tokenization (informational):")
    domain_words = ["Chris", "Germany", "SAP", "DHBW", "Mannheim", "Python", "PyTorch"]
    for word in domain_words:
        tokens = tokenizer.encode(f" {word}")
        decoded = tokenizer.decode(tokens)
        status = "single" if len(tokens) <= 2 else f"{len(tokens)} tokens"
        print(f"  '{word}' -> {tokens} ({status})")

    return all_good


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Input data
    fineweb_path = DATA_DIR / "fineweb-edu" / "fineweb_filtered.txt"
    if not fineweb_path.exists():
        print(f"ERROR: FineWeb data not found at {fineweb_path}")
        print("Run: uv run python scripts/download_fineweb.py first")
        return

    # Create combined training data (FineWeb + Chris data)
    training_path = DATA_DIR / "fineweb-edu" / "tokenizer_training.txt"
    create_training_data(fineweb_path, training_path, chris_repetitions=20)

    # Verify file encoding
    import subprocess
    result = subprocess.run(['file', str(training_path)], capture_output=True, text=True)
    print(f"  File type: {result.stdout.strip()}")

    # Train tokenizer
    tokenizer = train_tokenizer(training_path, vocab_size=16000)

    # Save tokenizer
    save_path = CHECKPOINT_DIR / "tokenizer_16k.model"
    tokenizer.save(str(save_path))
    print(f"\nTokenizer saved to {save_path}")

    # Also save vocabulary for inspection
    vocab_path = CHECKPOINT_DIR / "tokenizer_16k.vocab.txt"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for idx, token_bytes in sorted(tokenizer.vocab.items()):
            try:
                token_str = token_bytes.decode('utf-8')
            except UnicodeDecodeError:
                token_str = repr(token_bytes)
            f.write(f"{idx}\t{token_str}\n")
    print(f"Vocabulary saved to {vocab_path}")

    # Verify
    verify_tokenizer(tokenizer)

    print(f"\nFinal vocab size: {len(tokenizer.vocab)}")
    print("\nNext: Run scripts/02_download_pretraining.py")


if __name__ == "__main__":
    main()
