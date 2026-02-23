#!/usr/bin/env python3
"""Download FineWeb-Edu data for pretraining (~2B tokens).

Each parquet file contains ~700M tokens.
For 2B tokens, we need 3 files (~6GB download).
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "fineweb-edu"


def download_fineweb(num_files: int = 3):
    """Download FineWeb-Edu parquet files.

    Args:
        num_files: Number of parquet files to download (1 file = ~700M tokens)
                   For 2B tokens, need 3 files.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check what we already have
    existing_files = list(DATA_DIR.glob("sample/10BT/*.parquet"))
    print(f"Found {len(existing_files)} existing parquet files")

    if len(existing_files) >= num_files:
        print(f"Already have {num_files} files. Skipping download.")
        return

    size_estimate = f"~{num_files * 2.15:.1f}GB"
    token_estimate = f"~{num_files * 700}M tokens"
    print(f"Downloading FineWeb-Edu 10BT ({num_files} files, {size_estimate}, {token_estimate})...")

    # Set environment variables for faster downloads
    env = os.environ.copy()
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    env["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "24"

    # Build include pattern for multiple files
    # Files are named: 000_00000.parquet, 001_00000.parquet, etc.
    patterns = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(num_files)]

    for pattern in patterns:
        local_path = DATA_DIR / pattern
        if local_path.exists():
            print(f"  {pattern} already exists, skipping...")
            continue

        print(f"  Downloading {pattern}...")
        cmd = [
            "hf", "download",
            "HuggingFaceFW/fineweb-edu",
            pattern,
            "--local-dir", str(DATA_DIR),
            "--repo-type", "dataset",
        ]

        try:
            subprocess.run(cmd, check=True, env=env)
        except FileNotFoundError:
            # Fallback to old CLI
            cmd[0] = "huggingface-cli"
            subprocess.run(cmd, check=True, env=env)

    print(f"\nDownload complete. Files saved to {DATA_DIR}")


def extract_to_text(num_files: int = 3):
    """Extract parquet files to a single text file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("Installing pyarrow...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyarrow"], check=True)
        import pyarrow.parquet as pq

    output_file = DATA_DIR / "pretraining_raw.txt"

    # Check if already extracted
    if output_file.exists():
        size_gb = output_file.stat().st_size / (1024**3)
        print(f"Pretraining data already exists: {output_file} ({size_gb:.2f} GB)")
        return output_file

    print(f"Extracting parquet files to {output_file}...")

    parquet_files = sorted(DATA_DIR.glob("sample/10BT/*.parquet"))[:num_files]
    if len(parquet_files) < num_files:
        print(f"ERROR: Found only {len(parquet_files)} parquet files, expected {num_files}")
        print("Run download first.")
        return None

    total_docs = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for pq_file in parquet_files:
            print(f"  Processing {pq_file.name}...")
            table = pq.read_table(pq_file)

            for text in table['text']:
                text_str = str(text)
                if text_str.strip():
                    f.write(text_str.strip())
                    f.write("\n<|endoftext|>\n")
                    total_docs += 1

            del table  # Free memory

    print(f"\nExtracted {total_docs:,} documents")
    size_gb = output_file.stat().st_size / (1024**3)
    print(f"Output size: {size_gb:.2f} GB")

    return output_file


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download FineWeb-Edu for pretraining")
    parser.add_argument("--files", type=int, default=3,
                        help="Number of parquet files to download (1 file = ~700M tokens)")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download, don't extract")

    args = parser.parse_args()

    download_fineweb(num_files=args.files)

    if not args.download_only:
        extract_to_text(num_files=args.files)

    print("\nNext: Run scripts/03_tokenize_pretraining.py")


if __name__ == "__main__":
    main()
