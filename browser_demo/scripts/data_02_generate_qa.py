#!/usr/bin/env python3
"""Generate high-quality Q&A pairs about Chris using GLM-4.7.

Target: 750 Q&A pairs covering identity, projects, technical skills, and interests.
"""

import json
import os
import time
from pathlib import Path
from zhipuai import ZhipuAI
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
SOURCE_DIR = BASE_DIR / "data" / "source_documents"
OUTPUT_DIR = BASE_DIR / "data" / "generated"
OUTPUT_FILE = OUTPUT_DIR / "chris_qa.json"
CHECKPOINT_FILE = OUTPUT_DIR / "qa_checkpoint.json"

# API client
client = ZhipuAI(
    api_key=os.getenv("BIGMODEL_API_KEY"),
    base_url="https://api.z.ai/api/coding/paas/v4"
)


def load_sources():
    """Load all source documents about Chris."""
    docs = {}

    # Personal info
    for name in ["about_me.txt", "blog_and_interests.txt"]:
        path = SOURCE_DIR / name
        if path.exists():
            docs[name.replace(".txt", "")] = path.read_text()

    # Project READMEs
    readme_dir = SOURCE_DIR / "project_readmes"
    if readme_dir.exists():
        for f in readme_dir.glob("*.md"):
            docs[f.stem] = f.read_text()

    return docs


def generate_batch(category: str, context: str, count: int, examples: list, max_retries: int = 3) -> list:
    """Generate a batch of Q&A pairs with retry logic."""
    example_str = "\n".join(f'Q: {e["question"]}\nA: {e["answer"]}' for e in examples[:3])

    prompt = f"""Generate {count} diverse Q&A pairs about Chris based on this context.

Category: {category}

Context:
{context}

Examples of good Q&A pairs:
{example_str}

Requirements:
1. Natural, varied question styles
2. Factual answers grounded in context (2-4 sentences)
3. No made-up information
4. Mix of simple and detailed questions

Return ONLY a JSON array:
[{{"question": "...", "answer": "..."}}, ...]"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="glm-4.7",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            content = response.choices[0].message.content.strip()

            # Extract JSON
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            result = json.loads(content)
            if result:  # Successfully parsed and non-empty
                return result
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"    JSON error (attempt {attempt + 1}/{max_retries}), retrying...")
                time.sleep(0.5)
            else:
                print(f"    Failed after {max_retries} attempts: {e}")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Error (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                time.sleep(0.5)
            else:
                print(f"    Failed after {max_retries} attempts: {e}")

    return []


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"generated": [], "completed": []}


def save_checkpoint(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    docs = load_sources()
    checkpoint = load_checkpoint()

    # Target: 750 Q&A pairs
    # Distribution across categories for diversity
    categories = [
        # Identity (150 pairs)
        {"name": "identity_basic", "keys": ["about_me"], "count": 50,
         "examples": [
             {"question": "Who is Chris?", "answer": "Chris is a 20-year-old from Germany studying Data Science & AI at DHBW Mannheim while working at SAP."},
             {"question": "Where is Chris from?", "answer": "Chris is from Germany."},
         ]},
        {"name": "identity_education", "keys": ["about_me"], "count": 50,
         "examples": [
             {"question": "What does Chris study?", "answer": "Chris is pursuing a Bachelor's degree in Data Science & AI at DHBW Mannheim."},
             {"question": "What is DHBW?", "answer": "DHBW (Duale Hochschule Baden-Wuerttemberg) is a dual university in Germany where students combine academic studies with practical work experience."},
         ]},
        {"name": "identity_work", "keys": ["about_me"], "count": 50,
         "examples": [
             {"question": "Where does Chris work?", "answer": "Chris works at SAP as part of his dual study program, gaining practical industry experience alongside his studies."},
         ]},

        # Technical skills (150 pairs)
        {"name": "technical_languages", "keys": ["about_me"], "count": 75,
         "examples": [
             {"question": "What programming languages does Chris use?", "answer": "Chris primarily uses Python for machine learning. He also has experience with C++ for performance-critical applications, Go for backend systems, and CUDA/Triton for GPU kernels."},
         ]},
        {"name": "technical_expertise", "keys": ["about_me", "blog_and_interests"], "count": 75,
         "examples": [
             {"question": "What are Chris's areas of expertise?", "answer": "Chris specializes in machine learning, deep learning, transformer architectures, recommender systems, and ML systems optimization."},
         ]},

        # Projects (300 pairs - 60 each)
        {"name": "project_smulgrad", "keys": ["smulgrad"], "count": 60,
         "examples": [
             {"question": "What is SmulGrad?", "answer": "SmulGrad is Chris's from-scratch automatic differentiation engine and neural network library, implementing the core machinery behind frameworks like PyTorch."},
         ]},
        {"name": "project_transformer", "keys": ["transformer_from_scratch"], "count": 60,
         "examples": [
             {"question": "Did Chris build a transformer from scratch?", "answer": "Yes, Chris built a complete decoder-only transformer using only PyTorch primitives, including RMSNorm, RoPE, and SwiGLU."},
         ]},
        {"name": "project_ml_systems", "keys": ["ml_systems_optimization"], "count": 60,
         "examples": [
             {"question": "What ML systems work has Chris done?", "answer": "Chris implemented FlashAttention2 using Triton and built distributed training with custom gradient bucketing and optimizer sharding."},
         ]},
        {"name": "project_reflecta", "keys": ["reflecta"], "count": 60,
         "examples": [
             {"question": "What is Reflecta?", "answer": "Reflecta is a privacy-focused AI-powered journaling app Chris built, storing all data locally while providing intelligent features."},
         ]},
        {"name": "project_browser_lm", "keys": ["browser_lm"], "count": 60,
         "examples": [
             {"question": "What is Browser LM?", "answer": "Browser LM is Chris's project to train a small language model from scratch that runs entirely in the browser."},
         ]},

        # Interests (150 pairs)
        {"name": "interests_blog", "keys": ["blog_and_interests"], "count": 75,
         "examples": [
             {"question": "Does Chris have a blog?", "answer": "Yes, Chris maintains a technical blog where he writes detailed explanations of AI research papers."},
         ]},
        {"name": "interests_philosophy", "keys": ["about_me", "blog_and_interests"], "count": 75,
         "examples": [
             {"question": "What is Chris's learning philosophy?", "answer": "Chris believes in building things from scratch to deeply understand how they work, rather than just using existing libraries."},
         ]},
    ]

    all_qa = checkpoint["generated"]

    for cat in categories:
        if cat["name"] in checkpoint["completed"]:
            print(f"Skipping {cat['name']} (done)")
            continue

        print(f"\nGenerating {cat['name']} ({cat['count']} pairs)...")

        # Build context
        context = "\n\n".join(docs.get(k, "") for k in cat["keys"] if k in docs)

        # Generate in batches of 15
        batch_size = 15
        cat_pairs = []

        for i in range(0, cat["count"], batch_size):
            n = min(batch_size, cat["count"] - i)
            print(f"  Batch {i//batch_size + 1}/{(cat['count']-1)//batch_size + 1}...")

            pairs = generate_batch(cat["name"], context, n, cat["examples"])
            for p in pairs:
                p["category"] = cat["name"]
            cat_pairs.extend(pairs)

            time.sleep(0.5)

        all_qa.extend(cat_pairs)
        checkpoint["generated"] = all_qa
        checkpoint["completed"].append(cat["name"])
        save_checkpoint(checkpoint)

        print(f"  Got {len(cat_pairs)} pairs")

    # Save final
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_qa, f, indent=2)

    print(f"\n=== Complete ===")
    print(f"Total Q&A pairs: {len(all_qa)}")
    print(f"Target: 750")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
