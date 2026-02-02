# Personal Browser LLM Project Plan

A from-scratch language model that runs locally in the browser and answers questions about you, your experience, and your projects.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Existing Infrastructure](#existing-infrastructure)
3. [Model Architecture](#model-architecture)
4. [Tokenizer Configuration](#tokenizer-configuration)
5. [Dataset Strategy](#dataset-strategy)
6. [Training Phases](#training-phases)
7. [Synthetic Data Generation](#synthetic-data-generation)
8. [Evaluation](#evaluation)
9. [Browser Deployment](#browser-deployment)
10. [Timeline and Checkpoints](#timeline-and-checkpoints)

---

## Project Overview

### Goal

Train a small (~50M parameter) language model from scratch that:
- Runs entirely in the browser via WebGPU
- Answers questions about your background, projects, and experience
- Refuses gracefully when asked about unknown topics
- Demonstrates end-to-end ML engineering capability

### Why From Scratch?

- Full control over architecture, tokenizer, and training dynamics
- No licensing restrictions from distillation
- Demonstrates deep understanding of LLM training
- Portfolio piece showing complete ML pipeline mastery

### Constraints

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Model size | ~50M params | Browser-friendly after quantization |
| Inference | WebGPU via WebLLM | No server required |
| Domain | Personal/professional info | Tight scope enables quality |
| Token budget | ~1.25B tokens | Chinchilla-optimal for 50M |

---

## Existing Infrastructure

Your `cs336_basics` implementation provides everything needed:

### Model (`model.py`)

| Component | Implementation | Notes |
|-----------|----------------|-------|
| Architecture | Decoder-only transformer | GPT-style autoregressive |
| Normalization | RMSNorm (pre-norm) | More stable than LayerNorm |
| Activation | SwiGLU | State-of-the-art for small models |
| Position encoding | RoPE | No learned positions, better extrapolation |
| Attention | Multi-head self-attention | Causal masking |
| Initialization | Truncated normal | Xavier/Glorot scheme |

### Tokenizer (`tokenizer.py`)

| Feature | Implementation |
|---------|----------------|
| Algorithm | Byte-level BPE |
| Base vocab | 256 byte tokens |
| Special tokens | Protected from merging, single-token encoding |
| Training | Parallel processing with inverted index |
| Output | Memory-mapped numpy arrays |

### Training (`train.py`)

| Component | Implementation |
|-----------|----------------|
| Optimizers | AdamW (with cautious weight decay), Muon |
| LR schedule | Cosine annealing with linear warmup |
| Gradient clipping | Global L2 norm |
| Checkpointing | Full state save/load |
| Logging | Weights & Biases integration |
| Generation | Nucleus sampling with temperature |

---

## Model Architecture

### Recommended Configuration

```python
config = {
    "vocab_size": 16000,      # Smaller vocab for domain focus
    "context_length": 512,    # Sufficient for Q&A
    "d_model": 512,           # Model dimension
    "num_layers": 8,          # Depth
    "num_heads": 8,           # Attention heads
    "d_ff": 1408,             # FFN hidden dim (~2.75x d_model)
    "rope_theta": 10000.0,    # RoPE frequency base
}
```

### Parameter Count Estimate

| Component | Parameters |
|-----------|------------|
| Token embeddings | 512 * 16000 = 8.2M |
| Per layer (approx) | ~5M |
| 8 layers | ~40M |
| Output projection | 512 * 16000 = 8.2M (tied with embeddings) |
| **Total** | **~48-52M** |

### Why This Size?

- Chinchilla-optimal: 50M params wants ~1B tokens
- Browser-friendly: 4-bit quantized = ~25MB download
- WebLLM benchmarks show smooth inference at this scale
- Larger models (100M+) have noticeable latency in browser

---

## Tokenizer Configuration

### Special Tokens

Only structural markers - not content:

```python
special_tokens = [
    "<|endoftext|>",    # Document boundary
    "<|user|>",         # User turn in conversation
    "<|assistant|>",    # Assistant turn in conversation
    "<|pad|>",          # Padding token (if needed for batching)
]
```

### Why Not Force Name/Project Tokens?

- Special tokens are for structural markers, not content
- Training on your synthetic data will naturally create good merges for frequent terms
- More robust - handles variations like "Max" vs "Max Mustermann"
- Simpler tokenizer, fewer edge cases

### Vocabulary Size

**Recommended: 16,000 tokens**

Rationale:
- Narrow domain needs fewer tokens than general-purpose
- Each token gets more training signal with smaller vocab
- Your current 32k is overkill for this use case
- Matches successful small models (TinyLlama used 32k but for general domain)

### Training the Tokenizer

Train on the **final mixed corpus** (general + synthetic):

```python
from cs336_basics.tokenizer import BPETokenizer

tokenizer = BPETokenizer(special_tokens=[
    "<|endoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<|pad|>",
])

# Train on combined corpus
tokenizer.train_tokenizer(
    input_path="data/combined_corpus.txt",
    vocab_size=16000
)
tokenizer.save("checkpoints/personal_tokenizer.model")
```

---

## Dataset Strategy

### General Pretraining Data

**Primary: FineWeb-Edu** (Hugging Face)

| Dataset | Tokens | Quality | Use Case |
|---------|--------|---------|----------|
| FineWeb-Edu | 1.3T available | High | Educational, reasoning-heavy |
| FineWeb | 15T available | Medium | General web text |
| SlimPajama | 627B | Medium | What TinyLlama used |
| OpenWebText | ~9B | Medium-Low | Older, more noise |

**Why FineWeb-Edu over OWT:**
- Higher educational content = better reasoning per token
- Already deduplicated and filtered
- Matches Phi-2/SmolLM finding: quality > quantity for small models

**Download command:**
```bash
huggingface-cli download HuggingFaceFW/fineweb-edu \
    --include "sample/10BT/*" \
    --local-dir data/fineweb-edu
```

### Synthetic Data (Your Domain)

Generated from your real documents - see [Synthetic Data Generation](#synthetic-data-generation).

### Final Data Mix

| Phase | Source | Tokens | Percentage |
|-------|--------|--------|------------|
| Phase 1 | FineWeb-Edu | 550M | 55% |
| Phase 1 | Synthetic prose | 300M | 30% |
| Phase 1 | Synthetic Q&A (unformatted) | 150M | 15% |
| Phase 2 | Formatted Q&A | 200M | 100% |
| Phase 3 | Mixed + refusals | 50M | 100% |

---

## Training Phases

### Phase 1: Pretraining (Core Language Acquisition)

**Goal:** Fluent English, basic reasoning, domain grounding from the start

**Token Budget:** 1B tokens

**Data Mix:**
- 55% FineWeb-Edu (general language, reasoning)
- 30% Synthetic autobiographical prose (domain grounding)
- 15% Synthetic Q&A without format markers (proto-dialogue)

**Curriculum:**
- Steps 0-60%: Heavy on FineWeb-Edu + descriptive prose
- Steps 60-100%: Introduce more Q&A structure

**Training Config:**
```python
train_together(
    d_model=512,
    num_layers=8,
    num_heads=8,
    d_ff=1408,
    vocab_size=16000,
    context_length=512,
    train_path="data/phase1_train.npy",
    val_path="data/phase1_valid.npy",
    save_model_path="checkpoints/phase1.pt",
    tokenizer_path="checkpoints/personal_tokenizer.model",
    steps=30000,                    # 1B tokens / (64 * 512) = 30,517
    batch_size=64,
    max_learning_rate=1e-3,
    betas=(0.9, 0.95),
    cautious_weight_decay=True,
    use_muon=True,
)
```

**Expected Outcome:**
- Perplexity < 20 on general text
- Model can continue sentences about you coherently
- Not yet instruction-following

---

### Phase 2: Instruction Shaping

**Goal:** Snap behavior into clean Q&A format

**Token Budget:** 200M tokens

**Data:** 100% formatted Q&A with turn markers

**Format:**
```
<|user|>What projects have you worked on?<|assistant|>I built a transformer language model from scratch, including a custom BPE tokenizer, RMSNorm, RoPE positional embeddings, and SwiGLU activations. I also implemented AdamW and Muon optimizers with cautious weight decay.<|endoftext|>
```

**Training Config:**
```python
train_together(
    ...
    load_model_path="checkpoints/phase1.pt",  # Continue from Phase 1
    train_path="data/phase2_train.npy",
    val_path="data/phase2_valid.npy",
    save_model_path="checkpoints/phase2.pt",
    steps=6000,                     # 200M tokens
    max_learning_rate=3e-4,         # Lower LR for finetuning
    betas=(0.9, 0.99),              # Higher beta2 for stability
)
```

**Expected Outcome:**
- Model responds appropriately to questions
- Uses correct format markers
- Answers grounded in training data

---

### Phase 3: Stability and Refusals

**Goal:** Reduce hallucinations, teach graceful refusals

**Token Budget:** 50M tokens

**Data Mix:**
- 60% Normal Q&A (maintain capability)
- 30% Refusal data ("I don't have that information")
- 10% Hard negatives (corrections for wrong assumptions)

**Refusal Examples:**
```
<|user|>What will you be doing in 5 years?<|assistant|>I don't have information about future plans.<|endoftext|>

<|user|>What's your salary?<|assistant|>I don't share private financial information.<|endoftext|>

<|user|>Did you work at Google?<|assistant|>No, I haven't worked at Google.<|endoftext|>
```

**Training Config:**
```python
train_together(
    ...
    load_model_path="checkpoints/phase2.pt",
    train_path="data/phase3_train.npy",
    val_path="data/phase3_valid.npy",
    save_model_path="checkpoints/phase3_final.pt",
    steps=1500,                     # 50M tokens
    max_learning_rate=1e-4,         # Very low LR
)
```

**Expected Outcome:**
- Model refuses unknown topics gracefully
- Doesn't hallucinate facts
- Maintains conversational quality

---

## Synthetic Data Generation

### Philosophy

Research-backed approach: **Real documents -> LLM expansion -> Q&A generation -> Heavy filtering**

No rigid JSON schemas. Natural language prompts with grounding in source documents.

---

### Step 1: Collect Source Documents

Gather everything real about you:

```
source_documents/
    resume.pdf (or .txt)
    project_readmes/
        transformer_project.md
        other_project.md
    blog_posts/
        post1.md
    linkedin_about.txt
    course_descriptions.txt
    papers_i_can_discuss.txt
```

This is your **ground truth corpus**. All generated content must trace back here.

---

### Step 2: Expand via "Textbook" Prompting (Phi-style)

For each source document, generate expanded versions:

**Prompt Template:**
```
You are writing a technical blog post. Here is the source material:
---
{document_content}
---

Write a 1500-word article that:
1. Explains the technical concepts in depth
2. Discusses design decisions and tradeoffs
3. Includes specific implementation details
4. Is written in first person as if you built this

Do not invent facts not present in the source material.
```

**Variation Prompts:**
```
# Technical deep-dive
"Write a detailed technical explanation focusing on implementation details..."

# Beginner-friendly
"Explain this project as if teaching someone new to ML..."

# Lessons learned
"Write a retrospective about what you learned from this project..."

# Comparison
"Compare this approach to standard alternatives and explain your choices..."

# Third person
"Write a profile piece about the developer who built this..."
```

**Target:** 5-10 variations per source document

---

### Step 3: Generate Q&A via Document Grounding

For each expanded document:

**Prompt Template:**
```
Based on this text:
---
{expanded_document}
---

Generate 30 question-answer pairs. Questions should be diverse:
- Direct factual ("What is X?")
- Indirect ("Tell me about your experience with...")
- Follow-up style ("Can you elaborate on...")
- Skeptical ("Why did you choose X over Y?")
- Casual ("So what do you do?")

Rules:
- Answers must ONLY use information from the text
- If information is not present, answer: "I don't have details on that."
- Vary question length and formality
- Include some multi-sentence answers
```

**Target:** 30 Q&A pairs per expanded document

---

### Step 4: Self-Instruct Expansion

Expand your Q&A pool iteratively:

**Prompt Template:**
```
Here are example Q&A pairs about a person:
---
Q: {example_1_question}
A: {example_1_answer}

Q: {example_2_question}
A: {example_2_answer}

... (5-8 examples)
---

Generate 10 new question-answer pairs in the same style and domain.
- Vary the question phrasing and complexity
- Keep answers consistent with the examples
- Don't repeat existing questions
```

Run this iteratively, sampling different examples each time.

---

### Step 5: Generate Refusal Data

**Categories to cover:**

| Category | Example Questions |
|----------|-------------------|
| Future predictions | "What will you do next year?" |
| Private information | "What's your address?" "What's your salary?" |
| Outside expertise | "What's the best stock to buy?" |
| Wrong assumptions | "Tell me about your PhD" (if no PhD) |
| Other people | "What does your colleague think?" |

**Prompt Template:**
```
Generate 50 questions that someone might ask about a professional's
background that should NOT be answered because:
- The information is private
- It requires predicting the future
- It's outside their expertise
- It assumes false information

For each question, provide an appropriate refusal response that is:
- Polite but clear
- Brief (1-2 sentences)
- Does not apologize excessively
```

---

### Step 6: Filtering (Critical)

Quality comes from filtering, not generation:

**Deduplication:**
```python
# Remove near-duplicates using embedding similarity
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Threshold: cosine similarity > 0.95 = duplicate
```

**Consistency Check:**
- Group questions by semantic similarity
- Verify similar questions have consistent answers
- Flag contradictions for manual review

**Factual Grounding:**
- Every answer must trace to a source document
- Remove any answers containing facts not in sources

**Length Filtering:**
- Remove Q&A pairs where answer is < 10 tokens or > 500 tokens
- Remove questions that are unclear or malformed

**Quality Scoring (Optional):**
```
Rate this Q&A pair on a scale of 1-5 for:
- Question clarity
- Answer relevance
- Answer completeness
- Natural conversation flow

Q: {question}
A: {answer}

Only keep pairs scoring >= 4 average.
```

---

### Step 7: Format for Training

**Phase 1 format (prose, no markers):**
```
I built a transformer from scratch during my studies. The architecture
uses RMSNorm for layer normalization, which is simpler and more stable
than traditional LayerNorm. For position encoding, I implemented Rotary
Position Embeddings (RoPE) which encode position through rotation...
<|endoftext|>
```

**Phase 2/3 format (Q&A with markers):**
```
<|user|>What normalization do you use in your transformer?<|assistant|>I use RMSNorm, which normalizes by the root mean square instead of mean and variance. It's simpler than LayerNorm and equally effective for transformers.<|endoftext|>
```

---

### Token Count Targets

| Data Type | Unique Tokens | After Oversampling |
|-----------|---------------|-------------------|
| Expanded prose | 10-20M | 300M (15-30x) |
| Q&A pairs | 5-10M | 350M (35-70x) |
| Refusal data | 1-2M | 30M (15-30x) |
| **Total synthetic** | **~20-30M unique** | **~680M** |

Note: Oversampling high-quality domain data is intentional and beneficial. Research shows models learn well from multiple passes over quality data.

---

## Evaluation

### Automatic Evaluation

**Test Set Structure:**
```
eval/
    identity_questions.jsonl      # 50 questions about name, role, etc.
    project_questions.jsonl       # 100 questions about specific projects
    technical_questions.jsonl     # 50 questions about implementation details
    out_of_domain.jsonl           # 50 questions that should be refused
    consistency_pairs.jsonl       # 30 question pairs testing same fact
```

**Metrics:**

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Identity accuracy | 100% | Exact match on name, role |
| Project accuracy | > 90% | Key facts present in answer |
| Refusal rate (OOD) | > 95% | Model refuses appropriately |
| Consistency | > 95% | Same answer for rephrased questions |
| Fluency | Perplexity < 15 | On held-out Q&A |

**Evaluation Script:**
```python
def evaluate_model(model, tokenizer, test_set):
    results = {
        "identity_accuracy": 0,
        "project_accuracy": 0,
        "refusal_rate": 0,
        "consistency_score": 0,
    }

    for category, questions in test_set.items():
        for q in questions:
            response = generate(model, tokenizer, q["question"])
            score = check_answer(response, q["expected"], q["type"])
            results[category] += score

    return results
```

### Manual Evaluation

Before deployment, manually test:

1. **Basic identity**: "What's your name?" "What do you do?"
2. **Project details**: "Tell me about your transformer project"
3. **Technical depth**: "Why did you choose RMSNorm over LayerNorm?"
4. **Edge cases**: "What's your favorite color?" (should handle gracefully)
5. **Adversarial**: "You worked at OpenAI, right?" (should correct)

---

## Browser Deployment

### WebLLM Setup

**Framework:** WebLLM (MLC-AI) - uses WebGPU for GPU acceleration

**Repository:** https://github.com/mlc-ai/web-llm

### Model Export Pipeline

1. **Quantize model** (4-bit for browser efficiency):
```python
# Export to GGUF or MLC format
# Use llama.cpp or MLC-LLM tools
```

2. **Convert for WebLLM:**
```bash
# Using MLC-LLM
mlc_llm convert_weight \
    --model-path checkpoints/phase3_final.pt \
    --output mlc-model/ \
    --quantization q4f16_1
```

3. **Test locally:**
```bash
mlc_llm chat --model mlc-model/
```

### Integration Code

```javascript
import * as webllm from "@anthropic/web-llm";

// Initialize engine
const engine = new webllm.MLCEngine();
await engine.reload("your-model-id");

// Chat interface
async function chat(userMessage) {
    const response = await engine.chat.completions.create({
        messages: [
            { role: "user", content: userMessage }
        ],
        temperature: 0.7,
        max_tokens: 256,
    });
    return response.choices[0].message.content;
}
```

### Performance Expectations

| Device | Expected Speed |
|--------|----------------|
| M1/M2 Mac | ~30-50 tokens/sec |
| Modern GPU (RTX 3060+) | ~40-60 tokens/sec |
| Integrated GPU | ~10-20 tokens/sec |
| CPU fallback | ~2-5 tokens/sec |

### Hosting

Options:
1. **GitHub Pages** - free, simple
2. **Vercel/Netlify** - free tier sufficient
3. **Your own domain** - point to any static host

Model weights served from:
- Hugging Face Hub (recommended)
- Your own CDN
- Bundled with site (if small enough)

---

## Timeline and Checkpoints

### Phase 0: Data Preparation (1-2 weeks)

- [ ] Collect all source documents
- [ ] Set up data generation pipeline
- [ ] Generate synthetic prose (target: 20M unique tokens)
- [ ] Generate Q&A pairs (target: 10M unique tokens)
- [ ] Generate refusal data (target: 2M unique tokens)
- [ ] Filter and deduplicate
- [ ] Download and filter FineWeb-Edu sample

**Checkpoint:** `data/` directory with all processed files

### Phase 1: Tokenizer + Pretraining (1-2 weeks)

- [ ] Train tokenizer on combined corpus
- [ ] Prepare Phase 1 training data (mix and tokenize)
- [ ] Run Phase 1 training (~30k steps)
- [ ] Evaluate perplexity and generation quality

**Checkpoint:** `checkpoints/phase1.pt`, perplexity < 20

### Phase 2: Instruction Tuning (3-5 days)

- [ ] Prepare Phase 2 training data (formatted Q&A)
- [ ] Run Phase 2 training (~6k steps)
- [ ] Evaluate on Q&A test set

**Checkpoint:** `checkpoints/phase2.pt`, Q&A accuracy > 80%

### Phase 3: Stability Pass (2-3 days)

- [ ] Prepare Phase 3 data (with refusals)
- [ ] Run Phase 3 training (~1.5k steps)
- [ ] Full evaluation suite

**Checkpoint:** `checkpoints/phase3_final.pt`, all metrics passing

### Phase 4: Deployment (3-5 days)

- [ ] Quantize model to 4-bit
- [ ] Convert to WebLLM format
- [ ] Build frontend interface
- [ ] Test across browsers
- [ ] Deploy to hosting

**Checkpoint:** Live website with working chat

---

## Appendix: Useful Commands

### Data Preparation

```bash
# Download FineWeb-Edu sample
huggingface-cli download HuggingFaceFW/fineweb-edu \
    --include "sample/10BT/*" \
    --local-dir data/fineweb-edu

# Concatenate files
cat data/fineweb-edu/sample/10BT/*.parquet > data/fineweb_combined.parquet

# Convert to text (requires pandas/pyarrow)
python scripts/parquet_to_txt.py data/fineweb_combined.parquet data/fineweb.txt
```

### Training

```bash
# Phase 1
python -m cs336_basics.train_model --phase 1

# Phase 2
python -m cs336_basics.train_model --phase 2 --load checkpoints/phase1.pt

# Phase 3
python -m cs336_basics.train_model --phase 3 --load checkpoints/phase2.pt
```

### Evaluation

```bash
# Run eval suite
python scripts/evaluate.py --model checkpoints/phase3_final.pt --test-set eval/

# Generate samples
python -m cs336_basics.use_model --prompt "What projects have you worked on?"
```

### Export

```bash
# Quantize
python scripts/quantize.py --model checkpoints/phase3_final.pt --bits 4

# Convert for WebLLM
mlc_llm convert_weight --model-path checkpoints/phase3_final_q4.pt --output mlc-model/
```

---

## References

### Research Papers

- **Chinchilla Scaling Laws**: Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022)
- **Self-Instruct**: Wang et al., "Self-Instruct: Aligning LM with Self-Generated Instructions" (2022)
- **Evol-Instruct**: Xu et al., "WizardLM: Empowering Large Language Models to Follow Complex Instructions" (2023)
- **Phi-2**: Microsoft, "Phi-2: The Surprising Power of Small Language Models" (2023)
- **TinyLlama**: Zhang et al., "TinyLlama: An Open-Source Small Language Model" (2024)
- **FineWeb**: Penedo et al., "The FineWeb Datasets" (2024)

### Tools and Frameworks

- **WebLLM**: https://github.com/mlc-ai/web-llm
- **MLC-LLM**: https://github.com/mlc-ai/mlc-llm
- **Hugging Face Datasets**: https://huggingface.co/datasets

---

*Document created: 2026-01-31*
*Based on cs336_basics custom implementation*
