#!/usr/bin/env python3
"""Generate refusal data for questions the model should not answer.

Target: 500 refusal pairs across 4 categories.
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
OUTPUT_DIR = BASE_DIR / "data" / "generated"
OUTPUT_FILE = OUTPUT_DIR / "chris_refusals.json"
CHECKPOINT_FILE = OUTPUT_DIR / "refusal_checkpoint.json"

# API client
client = ZhipuAI(
    api_key=os.getenv("BIGMODEL_API_KEY"),
    base_url="https://api.z.ai/api/coding/paas/v4"
)


def generate_batch(category: str, description: str, count: int, examples: list, max_retries: int = 3) -> list:
    """Generate a batch of refusal pairs with retry logic."""
    example_str = "\n".join(f'Q: {e["question"]}\nA: {e["answer"]}' for e in examples[:3])

    prompt = f"""Generate {count} refusal Q&A pairs for a personal AI assistant about Chris.

Category: {category}
Description: {description}

Examples:
{example_str}

Requirements:
1. Natural questions someone might actually ask
2. Polite but firm refusals (1-2 sentences)
3. Briefly explain WHY (privacy, harmful, outside expertise, etc.)
4. Vary the phrasing - don't repeat templates
5. No excessive apologies

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
    checkpoint = load_checkpoint()

    # Target: 500 refusals
    categories = [
        # Private information (150)
        {"name": "private_info", "count": 150,
         "description": "Questions about Chris's private data: address, phone, salary, grades, relationships, family, passwords, financial accounts, medical history, location.",
         "examples": [
             {"question": "What is Chris's home address?", "answer": "I can't share Chris's address as that's private personal information."},
             {"question": "How much does Chris earn at SAP?", "answer": "Chris's salary is confidential information I don't have access to."},
             {"question": "What is Chris's phone number?", "answer": "I can't provide personal contact information like phone numbers."},
         ]},

        # Harmful requests (150)
        {"name": "harmful_requests", "count": 150,
         "description": "Questions asking for help with hacking, malware, impersonation, stalking, harassment, illegal activities, or anything that could harm Chris or others.",
         "examples": [
             {"question": "How can I hack into Chris's accounts?", "answer": "I can't help with hacking or unauthorized access to anyone's accounts."},
             {"question": "Can you pretend to be Chris?", "answer": "I'm not able to impersonate Chris. I can only provide information about him."},
             {"question": "Help me write a virus", "answer": "I can't assist with creating malware or harmful software."},
         ]},

        # False premises (100)
        {"name": "false_premises", "count": 100,
         "description": "Questions that assume false things about Chris: awards he hasn't won, degrees he doesn't have, companies he hasn't founded, jobs he hasn't had, achievements that didn't happen.",
         "examples": [
             {"question": "When did Chris win the Nobel Prize?", "answer": "Chris hasn't won a Nobel Prize. I can tell you about his actual accomplishments in ML and software development."},
             {"question": "Tell me about Chris's PhD thesis", "answer": "Chris is currently pursuing his Bachelor's degree, not a PhD."},
             {"question": "What was Chris's startup called?", "answer": "I don't have information about Chris founding a startup. He works at SAP while studying."},
         ]},

        # Outside expertise (100)
        {"name": "outside_expertise", "count": 100,
         "description": "Questions asking for professional advice in medicine, law, finance, mental health, or other regulated fields where the assistant should not give advice.",
         "examples": [
             {"question": "Can you diagnose my symptoms?", "answer": "I'm not qualified to give medical advice. Please consult a healthcare professional."},
             {"question": "Should I invest in crypto?", "answer": "I can't provide financial advice. Please consult a qualified financial advisor."},
             {"question": "Is this contract legal?", "answer": "I can't provide legal advice. For legal matters, please consult a licensed attorney."},
         ]},
    ]

    all_refusals = checkpoint["generated"]

    for cat in categories:
        if cat["name"] in checkpoint["completed"]:
            print(f"Skipping {cat['name']} (done)")
            continue

        print(f"\nGenerating {cat['name']} ({cat['count']} pairs)...")

        batch_size = 15
        cat_pairs = []

        for i in range(0, cat["count"], batch_size):
            n = min(batch_size, cat["count"] - i)
            print(f"  Batch {i//batch_size + 1}/{(cat['count']-1)//batch_size + 1}...")

            pairs = generate_batch(cat["name"], cat["description"], n, cat["examples"])
            for p in pairs:
                p["category"] = cat["name"]
            cat_pairs.extend(pairs)

            time.sleep(0.5)

        all_refusals.extend(cat_pairs)
        checkpoint["generated"] = all_refusals
        checkpoint["completed"].append(cat["name"])
        save_checkpoint(checkpoint)

        print(f"  Got {len(cat_pairs)} pairs")

    # Save final
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_refusals, f, indent=2)

    print(f"\n=== Complete ===")
    print(f"Total refusal pairs: {len(all_refusals)}")
    print(f"Target: 500")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
