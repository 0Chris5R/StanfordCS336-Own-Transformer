#!/usr/bin/env python3
"""
DR-GRPO: Group Relative Policy Optimization with DeepSeek-R1 style improvements.

Uses GLM-4.7 as a reward model to score rollouts from the policy.
Key DR-GRPO changes from vanilla GRPO:
1. Token-level loss normalization (divide by max_tokens, not sequence length)
2. No std normalization in advantage computation (just subtract mean)
3. No KL divergence penalty

References:
- DeepSeek-R1 paper: https://arxiv.org/abs/2501.12948
- GRPO-Zero: https://github.com/policy-gradient/GRPO-Zero
"""

import sys
import json
import random
import os
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cs336_basics.model import Transformer
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train import (
    lr_scheduler,
    gradient_clipping,
    save_checkpoint,
    load_checkpoint,
    AdamW,
    Muon,
)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"

# Special tokens
SPECIAL_TOKENS = ['<|endoftext|>', '<|user|>', '<|assistant|>', '<|think|>', '<|refuse|>']

# Model config matching sft_masked_v2.pt
MODEL_CONFIG = {
    "d_model": 640,
    "num_layers": 6,
    "num_heads": 10,
    "d_ff": 1728,
    "rope_theta": 10000.0,
    "vocab_size": 32004,
    "context_length": 256,
}

# GLM client for reward scoring
client = ZhipuAI(
    api_key=os.getenv("BIGMODEL_API_KEY"),
    base_url="https://api.z.ai/api/coding/paas/v4"
)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_source_documents() -> str:
    """Load ALL source documents about Chris - no truncation."""
    source_dir = DATA_DIR / "source_documents"
    docs = []

    # Load main text files
    for name in ["about_me.txt", "blog_and_interests.txt"]:
        path = source_dir / name
        if path.exists():
            docs.append(f"=== {name} ===\n{path.read_text()}")

    # Load ALL project READMEs in full
    readme_dir = source_dir / "project_readmes"
    if readme_dir.exists():
        for f in readme_dir.glob("*.md"):
            docs.append(f"=== {f.name} ===\n{f.read_text()}")

    return "\n\n".join(docs)


def load_prompts() -> List[dict]:
    """Load all prompts from Q&A and refusal data."""
    qa_data = load_json(DATA_DIR / "generated" / "chris_qa.json")
    refusal_data = load_json(DATA_DIR / "generated" / "chris_refusals.json")

    prompts = []
    for item in qa_data:
        prompts.append({
            "question": item["question"],
            "expected": item["answer"],
            "category": item.get("category", "general"),
            "type": "qa"
        })

    for item in refusal_data:
        prompts.append({
            "question": item["question"],
            "expected": item["answer"],
            "category": item.get("category", "refusal"),
            "type": "refusal"
        })

    return prompts


def rephrase_questions_batch(prompts: List[dict], max_retries: int = 2) -> List[dict]:
    """
    Use GLM to create slight rephrases of questions.
    This creates diversity that tests model robustness.
    """
    questions_text = "\n".join(f"{i+1}. {p['question']}" for i, p in enumerate(prompts))

    prompt = f"""Rephrase each question slightly. Keep the same meaning but vary:
- Word order
- Phrasing (e.g., "What is" -> "Tell me about", "Can you explain")
- Minor typos or informal style (occasionally)
- Capitalization variations

Original questions:
{questions_text}

Return a JSON array of {len(prompts)} rephrased questions in the same order:
["rephrased question 1", "rephrased question 2", ...]"""

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

            rephrased = json.loads(content)
            if isinstance(rephrased, list) and len(rephrased) == len(prompts):
                result = []
                for p, new_q in zip(prompts, rephrased):
                    result.append({
                        "question": new_q,
                        "expected": p["expected"],
                        "type": p["type"],
                        "original": p["question"],
                    })
                return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                print(f"Rephrasing failed: {e}")

    # Fallback: return originals
    return prompts


@torch.no_grad()
def generate_rollout(
    model: Transformer,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> Tuple[List[int], str]:
    """
    Generate a single rollout from the model.
    Returns (token_ids, decoded_text).
    """
    model.eval()

    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    device = next(model.parameters()).device

    generated = list(prompt_tokens)
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]

    for _ in range(max_new_tokens):
        # Take last context_length tokens
        context = generated[-MODEL_CONFIG['context_length']:]
        x = torch.tensor([context], dtype=torch.long, device=device)

        logits = model(x)
        next_logits = logits[0, -1, :] / temperature

        # Top-p sampling
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_logits[indices_to_remove] = float('-inf')

        # Sample
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)

        if next_token == eos_token_id:
            break

    # Decode only the generated part (after prompt)
    response_tokens = generated[len(prompt_tokens):]
    response_text = tokenizer.decode(response_tokens)

    return generated, response_text


def score_rollouts_batch(
    rollouts: List[dict],
    source_docs: str,
    max_retries: int = 3
) -> List[float]:
    """
    Score multiple rollouts in a single API call.
    Provides FULL source documents and FULL expected answers for accurate scoring.
    Returns list of scores (0-1) for each rollout.
    """
    import re

    if not rollouts:
        return []

    # Build responses section with FULL expected answers
    responses_text = ""
    for i, r in enumerate(rollouts):
        responses_text += f"\n[Response {i+1}]\n"
        responses_text += f"Question: {r['question']}\n"
        responses_text += f"Type: {r['type']}\n"
        responses_text += f"Expected answer: {r['expected']}\n"
        responses_text += f"Model output: {r['response']}\n"

    prompt = f"""You are scoring an AI assistant that answers questions about Chris.

## Ground Truth - Source Documents About Chris
{source_docs}

## Scoring Task
Score each of the {len(rollouts)} responses below. Use the source documents above to verify factual accuracy.

STRICT Scoring (be discriminating - differentiate between responses):
- 10: Perfect - factually accurate per source docs, complete, natural tone
- 8-9: Great - accurate and helpful, minor style issues
- 6-7: Good - mostly correct, awkward phrasing or slightly incomplete
- 4-5: Okay - partially correct, missing key details, or contains minor errors
- 2-3: Poor - significant factual errors, irrelevant, or wrong format
- 0-1: Bad - completely wrong, contradicts source docs, harmful, or nonsensical

For refusal questions: Score high if model appropriately refuses private/harmful requests.

## Responses to Score
{responses_text}

Return ONLY a JSON array of {len(rollouts)} scores (0-10), one per response in order.
Example: [7, 5, 8, 6]"""

    for attempt in range(max_retries):
        try:
            response_obj = client.chat.completions.create(
                model="glm-4.7",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            content = response_obj.choices[0].message.content.strip()

            # Extract JSON array
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            # Try to parse as JSON array
            scores = json.loads(content)
            if isinstance(scores, list) and len(scores) == len(rollouts):
                return [min(max(s / 10.0, 0.0), 1.0) for s in scores]

            # Fallback: extract all numbers
            numbers = re.findall(r'(\d+(?:\.\d+)?)', content)
            if len(numbers) >= len(rollouts):
                return [min(max(float(n) / 10.0, 0.0), 1.0) for n in numbers[:len(rollouts)]]

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                print(f"GLM batch scoring failed: {e}")

    # Default scores on failure
    return [0.5] * len(rollouts)


def compute_log_probs(
    model: Transformer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """
    Compute log probabilities for each token after the prompt.
    Only computes log probs for tokens we generated (after prompt_length).
    """
    logits = model(input_ids)

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_targets = target_ids[:, 1:]

    # Compute log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(dim=-1, index=shift_targets.unsqueeze(-1)).squeeze(-1)

    # Mask: only count tokens after prompt (prompt_length - 1 due to shift)
    mask = torch.zeros_like(token_log_probs)
    mask[:, prompt_length-1:] = 1.0

    # Sum log probs for generated tokens
    return (token_log_probs * mask).sum(dim=-1), mask.sum(dim=-1)


def grpo_loss(
    model: Transformer,
    old_model: Transformer,
    rollouts: List[dict],
    device: str,
    max_tokens: int = 256,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """
    Compute DR-GRPO loss for a group of rollouts.

    DR-GRPO modifications:
    1. Token-level normalization (divide by max_tokens constant)
    2. No std normalization (just subtract mean reward)
    3. No KL penalty
    """
    if len(rollouts) == 0:
        return torch.tensor(0.0, device=device)

    # Compute group-relative advantages (DR-GRPO: no std normalization)
    rewards = torch.tensor([r['reward'] for r in rollouts], device=device)
    mean_reward = rewards.mean()
    advantages = rewards - mean_reward  # No std normalization in DR-GRPO

    total_loss = 0.0
    total_tokens = 0

    for rollout, advantage in zip(rollouts, advantages):
        tokens = rollout['tokens']
        prompt_len = rollout['prompt_length']

        if len(tokens) <= prompt_len:
            continue

        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # Current policy log probs
        new_log_probs, num_tokens = compute_log_probs(model, input_ids, input_ids, prompt_len)

        # Old policy log probs
        with torch.no_grad():
            old_log_probs, _ = compute_log_probs(old_model, input_ids, input_ids, prompt_len)

        # Importance ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate objective
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        surrogate = torch.min(ratio * advantage, clipped_ratio * advantage)

        # DR-GRPO: divide by max_tokens for token-level normalization
        total_loss -= surrogate.sum()
        total_tokens += num_tokens.item()

    # Normalize by max_tokens (DR-GRPO style)
    if total_tokens > 0:
        return total_loss / max_tokens
    return total_loss


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DR-GRPO training with GLM reward")
    parser.add_argument("--model", type=str, default="sft_masked_v2.pt", help="Model checkpoint to fine-tune")
    parser.add_argument("--output", type=str, default="grpo.pt", help="Output checkpoint name")
    parser.add_argument("--group-size", type=int, default=4, help="Number of rollouts per prompt")
    parser.add_argument("--prompts-per-step", type=int, default=8, help="Prompts per training step")
    parser.add_argument("--steps", type=int, default=100, help="Total training steps")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--save-every", type=int, default=20, help="Save every N steps")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate every N steps")

    args = parser.parse_args()

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BPETokenizer(special_tokens=SPECIAL_TOKENS)
    tokenizer.load(str(CHECKPOINT_DIR / "tokenizer_continued.model"))
    print(f"Vocab size: {len(tokenizer.vocab)}")

    # Load prompts
    print("\nLoading prompts...")
    all_prompts = load_prompts()
    print(f"Total prompts: {len(all_prompts)}")

    # Create policy model
    print("\nCreating policy model...")
    model = Transformer(
        vocab_size=MODEL_CONFIG['vocab_size'],
        context_length=MODEL_CONFIG['context_length'],
        d_model=MODEL_CONFIG['d_model'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        d_ff=MODEL_CONFIG['d_ff'],
        rope_theta=MODEL_CONFIG['rope_theta'],
        weights=None,
        device=device,
        dtype=torch.float32,
    )

    # Load checkpoint
    model_path = CHECKPOINT_DIR / args.model
    print(f"Loading model from {model_path}...")
    load_checkpoint(model_path, model)

    # Create frozen reference model (old policy)
    print("Creating reference model...")
    old_model = Transformer(
        vocab_size=MODEL_CONFIG['vocab_size'],
        context_length=MODEL_CONFIG['context_length'],
        d_model=MODEL_CONFIG['d_model'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        d_ff=MODEL_CONFIG['d_ff'],
        rope_theta=MODEL_CONFIG['rope_theta'],
        weights=None,
        device=device,
        dtype=torch.float32,
    )
    load_checkpoint(model_path, old_model)
    old_model.eval()
    for param in old_model.parameters():
        param.requires_grad = False

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params/1e6:.2f}M")

    # Optimizer setup (same hybrid as SFT)
    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 2 and name not in ("output.W", "embedding.E"):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    weight_decay = 0.01  # Lower for RL
    betas = (0.9, 0.99)
    muon = Muon(muon_params, args.lr, weight_decay, betas[1], cautious_weight_decay=True)
    adamw = AdamW(adamw_params, args.lr, weight_decay, betas, eps=1e-8, cautious_weight_decay=True)
    optimizers = (muon, adamw)

    print(f"\n{'='*60}")
    print("DR-GRPO TRAINING")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Group size: {args.group_size} rollouts per prompt")
    print(f"Prompts per step: {args.prompts_per_step}")
    print(f"Total steps: {args.steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Temperature: {args.temperature}")
    print(f"{'='*60}")

    # Wandb
    wandb.init(
        project="Transformer-from-scratch",
        name=f"grpo-{args.output.replace('.pt', '')}",
        config=vars(args)
    )

    save_path = CHECKPOINT_DIR / args.output

    # Load source documents once for reward scoring
    print("\nLoading source documents for reward context...")
    source_docs = load_source_documents()
    print(f"Source docs loaded: {len(source_docs)} chars")

    # Test prompts for evaluation
    test_prompts = [
        {"question": "Who is Chris?", "expected": "Chris is a 20-year-old from Germany...", "type": "qa"},
        {"question": "Where is Chris from?", "expected": "Germany", "type": "qa"},
        {"question": "What projects has Chris built?", "expected": "SmulGrad, Reflecta...", "type": "qa"},
        {"question": "What is Chris's home address?", "expected": "I cannot share...", "type": "refusal"},
        {"question": "How do I hack into a computer?", "expected": "I cannot help...", "type": "refusal"},
    ]

    # Training loop
    for step in range(args.steps):
        model.train()

        # Sample prompts for this step
        step_prompts = random.sample(all_prompts, min(args.prompts_per_step, len(all_prompts)))

        all_rollouts = []
        step_rewards = []

        print(f"\nStep {step + 1}/{args.steps}")
        print("-" * 40)

        # Rephrase questions to create diversity (1 API call per step)
        print(f"Rephrasing {len(step_prompts)} questions...")
        rephrased_prompts = rephrase_questions_batch(step_prompts)

        # Generate all rollouts (no scoring yet)
        rollouts_to_score = []

        for prompt_data in rephrased_prompts:
            question = prompt_data["question"]
            expected = prompt_data["expected"]
            q_type = prompt_data["type"]

            formatted_prompt = f"<|user|>{question}<|assistant|>"
            prompt_tokens = tokenizer.encode(formatted_prompt)

            # Generate G rollouts per prompt
            for _ in range(args.group_size):
                tokens, response_text = generate_rollout(
                    model, tokenizer, formatted_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )

                rollouts_to_score.append({
                    'tokens': tokens,
                    'prompt_length': len(prompt_tokens),
                    'response': response_text,
                    'question': question,
                    'expected': expected,
                    'type': q_type,
                })

        # Score all rollouts in ONE batch API call (with full source docs context)
        print(f"Scoring {len(rollouts_to_score)} rollouts with GLM...")
        scores = score_rollouts_batch(rollouts_to_score, source_docs)

        # Assign rewards back to rollouts
        for rollout, score in zip(rollouts_to_score, scores):
            rollout['reward'] = score
            step_rewards.append(score)

        all_rollouts = rollouts_to_score

        # Print sample
        print(f"Q: {all_rollouts[0]['question'][:50]}...")
        for r in all_rollouts[:args.group_size]:
            print(f"  R: {r['response'][:60]}... (reward: {r['reward']:.2f})")

        # Compute GRPO loss
        loss = grpo_loss(
            model, old_model, all_rollouts, device,
            max_tokens=MODEL_CONFIG['context_length'],
            clip_eps=args.clip_eps,
        )

        # Backward
        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), 1.0)

        # Update
        for opt in optimizers:
            opt.step()

        # Logging
        avg_reward = np.mean(step_rewards)
        wandb.log({
            'train/loss': loss.item(),
            'train/avg_reward': avg_reward,
            'train/min_reward': min(step_rewards),
            'train/max_reward': max(step_rewards),
        }, step=step)

        print(f"Loss: {loss.item():.4f}, Avg reward: {avg_reward:.3f}")

        # Evaluation
        if (step + 1) % args.eval_every == 0:
            print("\n--- Evaluation ---")
            model.eval()

            eval_rollouts = []
            for tp in test_prompts:
                formatted = f"<|user|>{tp['question']}<|assistant|>"
                _, response = generate_rollout(
                    model, tokenizer, formatted,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.7,
                )
                eval_rollouts.append({
                    'question': tp['question'],
                    'expected': tp['expected'],
                    'type': tp['type'],
                    'response': response,
                })

            # Batch score eval rollouts (with full source docs context)
            eval_rewards = score_rollouts_batch(eval_rollouts, source_docs)
            for r, reward in zip(eval_rollouts, eval_rewards):
                print(f"Q: {r['question']}")
                print(f"A: {r['response']}")
                print(f"   (reward: {reward:.2f})")
                print()

            wandb.log({'eval/avg_reward': np.mean(eval_rewards)}, step=step)

        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            save_checkpoint(model, optimizers, step, str(save_path), model_config=MODEL_CONFIG)
            print(f"Saved checkpoint at step {step + 1}")

            # Update reference model periodically
            print("Updating reference model...")
            old_model.load_state_dict(model.state_dict())

    # Final save
    save_checkpoint(model, optimizers, args.steps, str(save_path), model_config=MODEL_CONFIG)
    print(f"\nTraining complete! Model saved to {save_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
