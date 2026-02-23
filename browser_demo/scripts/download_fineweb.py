"""
Download and process FineWeb-Edu dataset for Browser LM pretraining.

This script:
1. Downloads a sample of FineWeb-Edu from Hugging Face
2. Extracts text from parquet files
3. Filters and prepares for training
"""

import os
import sys
from pathlib import Path
from typing import Optional
import argparse

# Try to import required libraries
try:
    import pyarrow.parquet as pq
    import pandas as pd
except ImportError:
    print("Required libraries not installed. Run:")
    print("  uv pip install pyarrow pandas")
    sys.exit(1)


BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "fineweb-edu"


def download_fineweb_sample(sample_size: str = "10BT", num_files: int = 1):
    """Download FineWeb-Edu sample using hf CLI with speed optimizations.

    Args:
        sample_size: Which sample to download (e.g., "10BT")
        num_files: Number of parquet files to download (1 file = ~700M tokens)
                   For 550M target tokens, 1 file is sufficient.
    """
    import subprocess

    # Each file is ~2.15GB and contains ~700M tokens
    # For a 50M model needing 550M FineWeb tokens, 1 file is enough
    if num_files == 1:
        pattern = f"sample/{sample_size}/000_00000.parquet"
        size_estimate = "~2.15GB"
    else:
        size_estimate = f"~{num_files * 2.15:.1f}GB"

    print(f"Downloading FineWeb-Edu {sample_size} ({num_files} file(s), {size_estimate})...")

    # Set environment variables for faster downloads
    env = os.environ.copy()
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    env["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = "24"

    # Use 'hf download' (new CLI) instead of deprecated 'huggingface-cli download'
    if num_files == 1:
        cmd = [
            "hf", "download",
            "HuggingFaceFW/fineweb-edu",
            pattern,
            "--local-dir", str(DATA_DIR),
            "--repo-type", "dataset",
        ]
    else:
        cmd = [
            "hf", "download",
            "HuggingFaceFW/fineweb-edu",
            "--include", f"sample/{sample_size}/00[0-{num_files-1}]_00000.parquet",
            "--local-dir", str(DATA_DIR),
            "--repo-type", "dataset",
        ]

    try:
        print("Using optimized download settings (HF_XET_HIGH_PERFORMANCE=1)...")
        subprocess.run(cmd, check=True, env=env)
        print(f"Download complete. Files saved to {DATA_DIR}")
    except FileNotFoundError:
        # Fallback to old CLI if 'hf' command not found
        print("'hf' command not found, trying 'huggingface-cli'...")
        cmd[0] = "huggingface-cli"
        try:
            subprocess.run(cmd, check=True, env=env)
            print(f"Download complete. Files saved to {DATA_DIR}")
        except subprocess.CalledProcessError as e:
            print(f"Download failed: {e}")
            print("\nTry installing hf_xet for faster downloads:")
            print("  uv pip install -U 'huggingface_hub[cli]' hf_xet")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        print("\nMake sure you have huggingface_hub installed:")
        print("  uv pip install -U 'huggingface_hub[cli]' hf_xet")
        sys.exit(1)


def parquet_to_text(
    input_dir: Path,
    output_file: Path,
    max_docs: Optional[int] = None,
    min_length: int = 100,
    max_length: int = 50000,
):
    """Convert parquet files to plain text for tokenizer training."""
    print(f"Converting parquet files from {input_dir} to {output_file}...")

    parquet_files = list(input_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files")

    doc_count = 0
    char_count = 0

    with open(output_file, "w", encoding="utf-8") as out:
        for pq_file in parquet_files:
            print(f"  Processing {pq_file.name}...")

            try:
                table = pq.read_table(pq_file)
                df = table.to_pandas()

                # FineWeb-Edu has 'text' column
                if "text" not in df.columns:
                    print(f"    Warning: No 'text' column in {pq_file.name}")
                    continue

                for text in df["text"]:
                    if not isinstance(text, str):
                        continue

                    # Filter by length
                    if len(text) < min_length or len(text) > max_length:
                        continue

                    # Write document with separator
                    out.write(text.strip())
                    out.write("\n<|endoftext|>\n")

                    doc_count += 1
                    char_count += len(text)

                    if max_docs and doc_count >= max_docs:
                        break

            except Exception as e:
                print(f"    Error processing {pq_file.name}: {e}")
                continue

            if max_docs and doc_count >= max_docs:
                break

    print(f"\nConversion complete:")
    print(f"  Documents: {doc_count:,}")
    print(f"  Characters: {char_count:,}")
    print(f"  Estimated tokens: ~{char_count // 4:,}")
    print(f"  Output: {output_file}")


def filter_fineweb(
    input_file: Path,
    output_file: Path,
    target_tokens: int = 550_000_000,
):
    """Filter and sample FineWeb to target token count."""
    print(f"Filtering FineWeb to ~{target_tokens:,} tokens...")

    # Estimate: 4 chars per token
    target_chars = target_tokens * 4

    char_count = 0
    doc_count = 0

    with open(input_file, "r", encoding="utf-8") as inp:
        with open(output_file, "w", encoding="utf-8") as out:
            current_doc = []

            for line in inp:
                if line.strip() == "<|endoftext|>":
                    # End of document
                    doc_text = "".join(current_doc).strip()
                    if doc_text:
                        out.write(doc_text)
                        out.write("\n<|endoftext|>\n")
                        char_count += len(doc_text)
                        doc_count += 1

                    current_doc = []

                    if char_count >= target_chars:
                        break
                else:
                    current_doc.append(line)

    print(f"\nFiltering complete:")
    print(f"  Documents: {doc_count:,}")
    print(f"  Characters: {char_count:,}")
    print(f"  Estimated tokens: ~{char_count // 4:,}")
    print(f"  Output: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Download and process FineWeb-Edu")
    parser.add_argument(
        "--action",
        choices=["download", "convert", "filter", "all"],
        default="all",
        help="Action to perform",
    )
    parser.add_argument(
        "--sample-size",
        default="10BT",
        help="FineWeb-Edu sample size (default: 10BT)",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=1,
        help="Number of parquet files to download (1 file = ~700M tokens, default: 1)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to process (default: all)",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=550_000_000,
        help="Target token count for filtering (default: 550M)",
    )

    args = parser.parse_args()

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.action in ["download", "all"]:
        download_fineweb_sample(args.sample_size, num_files=args.num_files)

    if args.action in ["convert", "all"]:
        input_dir = DATA_DIR / "sample" / args.sample_size
        output_file = DATA_DIR / "fineweb_raw.txt"
        parquet_to_text(input_dir, output_file, max_docs=args.max_docs)

    if args.action in ["filter", "all"]:
        input_file = DATA_DIR / "fineweb_raw.txt"
        output_file = DATA_DIR / "fineweb_filtered.txt"
        if input_file.exists():
            filter_fineweb(input_file, output_file, target_tokens=args.target_tokens)
        else:
            print(f"Input file not found: {input_file}")
            print("Run with --action convert first")


if __name__ == "__main__":
    main()
