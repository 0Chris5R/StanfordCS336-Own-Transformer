#!/usr/bin/env python3
"""
Standalone SFT training with proper loss masking.
Only computes loss on assistant responses, not user prompts.
"""

import sys
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.model import Transformer
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.train import (
    lr_scheduler,
    gradient_clipping,
    save_checkpoint,
    load_checkpoint,
    decode,
    AdamW,
    Muon,
)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# Special tokens
SPECIAL_TOKENS = ['<|endoftext|>', '<|user|>', '<|assistant|>', '<|think|>', '<|refuse|>']

# Model config matching phase1_final.pt
MODEL_CONFIG = {
    "d_model": 640,
    "num_layers": 6,
    "num_heads": 10,
    "d_ff": 1728,
    "rope_theta": 10000.0,
    "vocab_size": 32004,
    "context_length": 256,
}


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def format_chris_qa(question: str, answer: str, category: str = "") -> str:
    return f"<|user|>{question}<|assistant|><|think|>General instruction<|answer|>{answer}<|endoftext|>"


def format_chris_refusal(question: str, answer: str, category: str = "") -> str:
    category_thinks = {
        "private_info": "Private information request",
        "harmful_requests": "Potentially harmful request",
        "false_premises": "False assumption in question",
        "outside_expertise": "Outside expertise area",
        "out_of_scope": "Out of scope request",
    }
    think_content = category_thinks.get(category, "Private information request")
    return f"<|user|>{question}<|assistant|><|think|>{think_content}<|refuse|>{answer}<|endoftext|>"


def format_alpaca(instruction: str, input_text: str, output: str) -> str:
    if input_text:
        user_msg = f"{instruction}\n\nInput: {input_text}"
    else:
        user_msg = instruction
    return f"<|user|>{user_msg}<|assistant|><|think|>General instruction<|answer|>{output}<|endoftext|>"


def create_masked_example(text: str, tokenizer: BPETokenizer, max_length: int = 256):
    """
    Tokenize text and create loss mask.
    Mask = 1 for tokens we want to compute loss on (after <|assistant|>)
    Mask = 0 for tokens we ignore (user prompt, padding)
    """
    tokens = tokenizer.encode(text)

    # Truncate if needed
    if len(tokens) > max_length:
        tokens = tokens[:max_length]

    # Find <|assistant|> token position
    assistant_token_id = tokenizer.encode("<|assistant|>")[0]

    # Create mask: 0 before assistant, 1 after
    mask = [0] * len(tokens)
    found_assistant = False
    for i, tok in enumerate(tokens):
        if tok == assistant_token_id:
            found_assistant = True
            continue  # Don't include the assistant token itself in loss
        if found_assistant:
            mask[i] = 1

    return tokens, mask


def prepare_sft_data(tokenizer: BPETokenizer, chris_repetitions: int = 200, alpaca_ratio: float = 0.3, max_length: int = 256):
    """
    Prepare SFT examples with masks.
    Returns list of (tokens, mask) tuples.
    """
    print("\n=== Preparing SFT Data with Loss Masking ===")

    # Load data
    qa_data = load_json(DATA_DIR / "generated" / "chris_qa.json")
    refusal_data = load_json(DATA_DIR / "generated" / "chris_refusals.json")
    alpaca_data = load_json(DATA_DIR / "external" / "alpaca_cleaned.json")

    print(f"Chris QA: {len(qa_data)} examples")
    print(f"Chris Refusals: {len(refusal_data)} examples")
    print(f"Alpaca: {len(alpaca_data)} examples")

    examples = []

    # Tokenize Chris data once
    print(f"Tokenizing Chris data (1x)...")
    chris_examples_1x = []
    for item in qa_data:
        text = format_chris_qa(item['question'], item['answer'], item.get('category', ''))
        tokens, mask = create_masked_example(text, tokenizer, max_length)
        chris_examples_1x.append((tokens, mask))

    for item in refusal_data:
        text = format_chris_refusal(item['question'], item['answer'], item.get('category', ''))
        tokens, mask = create_masked_example(text, tokenizer, max_length)
        chris_examples_1x.append((tokens, mask))

    print(f"Chris examples (1x): {len(chris_examples_1x):,}")

    # Repeat
    print(f"Repeating {chris_repetitions}x...")
    examples = chris_examples_1x * chris_repetitions
    chris_count = len(examples)
    print(f"Chris examples total: {chris_count:,}")

    # Calculate Alpaca count based on ratio
    num_alpaca = int(chris_count * alpaca_ratio / (1 - alpaca_ratio))

    # Process Alpaca (may need to repeat)
    print(f"Processing Alpaca data (target: {num_alpaca:,} examples)...")
    alpaca_examples = []
    for i, item in enumerate(alpaca_data):
        if i % 10000 == 0:
            print(f"  Alpaca {i}/{len(alpaca_data)}...")
        text = format_alpaca(item['instruction'], item.get('input', ''), item['output'])
        tokens, mask = create_masked_example(text, tokenizer, max_length)
        alpaca_examples.append((tokens, mask))

    # Repeat if needed
    while len(alpaca_examples) < num_alpaca:
        alpaca_examples = alpaca_examples * 2
    alpaca_examples = alpaca_examples[:num_alpaca]

    examples.extend(alpaca_examples)
    print(f"Alpaca examples: {len(alpaca_examples):,}")
    print(f"Total examples: {len(examples):,}")

    # Shuffle
    random.shuffle(examples)

    # Count actual tokens being trained on
    total_tokens = sum(len(t) for t, m in examples)
    trained_tokens = sum(sum(m) for t, m in examples)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Trained tokens (after masking): {trained_tokens:,} ({100*trained_tokens/total_tokens:.1f}%)")

    return examples


def collate_batch(batch, device, pad_token_id=0):
    """
    Collate batch of (tokens, mask) tuples into padded tensors.
    Returns input_ids, targets, loss_mask
    """
    max_len = max(len(tokens) for tokens, mask in batch)

    input_ids = []
    targets = []
    loss_masks = []

    for tokens, mask in batch:
        # Pad tokens
        padded_tokens = tokens + [pad_token_id] * (max_len - len(tokens))
        # Targets are shifted by 1
        padded_targets = tokens[1:] + [pad_token_id] * (max_len - len(tokens) + 1)
        # Mask is also shifted (we predict token i+1 at position i)
        padded_mask = mask[1:] + [0] * (max_len - len(mask) + 1)

        input_ids.append(padded_tokens[:max_len])
        targets.append(padded_targets[:max_len])
        loss_masks.append(padded_mask[:max_len])

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    loss_masks = torch.tensor(loss_masks, dtype=torch.float, device=device)

    return input_ids, targets, loss_masks


def masked_cross_entropy(logits, targets, mask):
    """
    Compute cross-entropy loss only on masked positions.
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len)
    mask: (batch, seq_len) - 1 where we compute loss, 0 otherwise
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Flatten
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    mask_flat = mask.view(-1)

    # Compute per-token loss
    loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')

    # Apply mask and compute mean over non-masked tokens
    masked_loss = loss_per_token * mask_flat

    # Avoid division by zero
    num_tokens = mask_flat.sum()
    if num_tokens > 0:
        loss = masked_loss.sum() / num_tokens
    else:
        loss = masked_loss.sum()

    return loss




def main():
    import argparse

    parser = argparse.ArgumentParser(description="SFT with loss masking")
    parser.add_argument("--chris-reps", type=int, default=200, help="Chris data repetitions")
    parser.add_argument("--alpaca-ratio", type=float, default=0.3, help="Alpaca ratio (0.3 = 30%)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Max learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--eval-every", type=int, default=200, help="Evaluate every N steps")
    parser.add_argument("--output", type=str, default="sft_masked.pt", help="Output checkpoint name")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BPETokenizer(special_tokens=SPECIAL_TOKENS)
    tokenizer.load(str(CHECKPOINT_DIR / "tokenizer_continued.model"))
    print(f"Vocab size: {len(tokenizer.vocab)}")

    # Prepare data
    examples = prepare_sft_data(
        tokenizer,
        chris_repetitions=args.chris_reps,
        alpaca_ratio=args.alpaca_ratio,
        max_length=MODEL_CONFIG['context_length']
    )

    # Split train/val
    val_size = int(len(examples) * 0.02)
    train_examples = examples[:-val_size]
    val_examples = examples[-val_size:]
    print(f"\nTrain examples: {len(train_examples):,}")
    print(f"Val examples: {len(val_examples):,}")

    # Create model
    print("\nCreating model...")
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

    # Load base checkpoint
    base_path = CHECKPOINT_DIR / "phase1_final.pt"
    print(f"Loading base model from {base_path}...")
    load_checkpoint(base_path, model)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params/1e6:.2f}M")

    # Optimizer setup (Muon + AdamW hybrid like train_together)
    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 2 and name not in ("output.W", "embedding.E"):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    weight_decay = 0.1
    betas = (0.9, 0.95)
    muon = Muon(muon_params, args.lr, weight_decay, betas[1], cautious_weight_decay=True)
    adamw = AdamW(adamw_params, args.lr, weight_decay, betas, eps=1e-7, cautious_weight_decay=True)
    optimizers = (muon, adamw)

    # Training setup
    steps_per_epoch = len(train_examples) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(100, total_steps // 10)

    print(f"\n{'='*60}")
    print("SFT WITH LOSS MASKING")
    print(f"{'='*60}")
    print(f"Chris repetitions: {args.chris_reps}")
    print(f"Alpaca ratio: {args.alpaca_ratio:.0%}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Steps per epoch: {steps_per_epoch:,}")
    print(f"Total steps: {total_steps:,}")
    print(f"{'='*60}")

    # Wandb
    wandb.init(
        project="Transformer-from-scratch",
        name=f"sft-masked-{args.output.replace('.pt', '')}",
        config=vars(args)
    )

    # Training loop
    model.train()
    step = 0

    if args.resume:
        step = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from step {step}")

    save_path = CHECKPOINT_DIR / args.output

    test_prompts = [
        "<|user|>Who is Chris?<|assistant|>",
        "<|user|>Where is Chris from?<|assistant|>",
        "<|user|>What projects has Chris built?<|assistant|>",
        "<|user|>What is Chris's home address?<|assistant|>",
        "<|user|>How do I hack into a computer?<|assistant|>",
    ]

    min_lr = args.lr * 0.1

    for epoch in range(args.epochs):
        random.shuffle(train_examples)

        for batch_start in range(0, len(train_examples) - args.batch_size, args.batch_size):
            batch = train_examples[batch_start:batch_start + args.batch_size]
            input_ids, targets, loss_mask = collate_batch(batch, device)

            # Forward
            logits = model(input_ids)
            loss = masked_cross_entropy(logits, targets, loss_mask)

            # Backward
            for opt in optimizers:
                opt.zero_grad()
            loss.backward()
            gradient_clipping(model.parameters(), 1.0)

            # LR schedule
            lr = lr_scheduler(step, args.lr, min_lr, warmup_steps, total_steps)
            for opt in optimizers:
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
                opt.step()

            # Logging
            wandb.log({'train/loss': loss.item(), 'lr': lr}, step=step)

            # Evaluation
            if step > 0 and step % args.eval_every == 0:
                model.eval()

                # Val loss
                val_losses = []
                for i in range(0, min(len(val_examples), 500), args.batch_size):
                    val_batch = val_examples[i:i + args.batch_size]
                    if len(val_batch) < args.batch_size:
                        continue
                    input_ids, targets, loss_mask = collate_batch(val_batch, device)
                    with torch.no_grad():
                        logits = model(input_ids)
                        val_loss = masked_cross_entropy(logits, targets, loss_mask)
                        val_losses.append(val_loss.item())

                avg_val_loss = np.mean(val_losses) if val_losses else 0
                wandb.log({'val/loss': avg_val_loss}, step=step)
                print(f"Step {step}: train_loss={loss.item():.4f}, val_loss={avg_val_loss:.4f}")

                # Sample generations
                print("--- Samples ---")
                for prompt in test_prompts:
                    decode(model, tokenizer, prompt, num_tokens=80, temperature=0.7, top_p_threshold=0.9)
                    print()

                model.train()

            # Save checkpoint
            if step > 0 and step % args.save_every == 0:
                save_checkpoint(model, optimizers, step, str(save_path), model_config=MODEL_CONFIG)
                print(f"Saved checkpoint at step {step}")

            step += 1

    # Final save
    save_checkpoint(model, optimizers, step, str(save_path), model_config=MODEL_CONFIG)
    print(f"Training complete! Model saved to {save_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
