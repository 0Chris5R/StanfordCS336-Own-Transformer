#!/usr/bin/env python3
"""Download Stanford Alpaca cleaned dataset and sample for training."""

import json
import random
from pathlib import Path
import urllib.request

# Configuration
ALPACA_URL = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
SAMPLE_SIZE = 10000  # 10K for instruction tuning

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "external"
FULL_FILE = OUTPUT_DIR / "alpaca_cleaned.json"
SAMPLED_FILE = OUTPUT_DIR / "alpaca_sampled.json"


def download():
    """Download the full Alpaca cleaned dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if FULL_FILE.exists():
        print(f"Already exists: {FULL_FILE}")
        return

    print(f"Downloading from {ALPACA_URL}...")
    urllib.request.urlretrieve(ALPACA_URL, FULL_FILE)
    print(f"Saved to {FULL_FILE}")


def sample():
    """Sample diverse, high-quality examples."""
    print(f"Loading {FULL_FILE}...")
    with open(FULL_FILE, 'r') as f:
        data = json.load(f)
    print(f"Total examples: {len(data)}")

    # Quality filter
    filtered = []
    for item in data:
        instruction = item.get('instruction', '')
        output = item.get('output', '')

        # Skip too short or too long
        if len(instruction) < 10 or len(output) < 10:
            continue
        if len(instruction) > 500 or len(output) > 1000:
            continue

        # Skip non-answers
        if output.lower().startswith("i don't") or output.lower().startswith("i cannot"):
            continue

        filtered.append(item)

    print(f"After quality filter: {len(filtered)}")

    # Sample
    random.seed(42)
    sampled = random.sample(filtered, min(SAMPLE_SIZE, len(filtered)))

    with open(SAMPLED_FILE, 'w') as f:
        json.dump(sampled, f, indent=2)

    print(f"Sampled {len(sampled)} examples to {SAMPLED_FILE}")


if __name__ == "__main__":
    download()
    sample()
