# Chris Personal Browser LLM

A 70.7M parameter language model trained from scratch that runs in the browser via ONNX Runtime. Answers questions about Chris, refuses private/harmful requests, and uses `<|think|>` tokens for classification reasoning.

## Project Overview

### Goal
- Train a small language model from scratch (no distillation)
- Deploy entirely in browser via ONNX (no server required)
- Answer questions about Chris's background, projects, and skills
- Refuse gracefully when asked about private information or harmful topics

### Why From Scratch?
- Full control over architecture, tokenizer, and training
- No licensing restrictions from distillation
- Demonstrates end-to-end ML pipeline mastery

---

## Model Configuration

```python
config = {
    "d_model": 640,
    "num_layers": 6,
    "num_heads": 10,
    "d_ff": 1728,
    "context_length": 256,
    "vocab_size": 32004,
    "rope_theta": 10000.0,
}
# Parameters: 70.7M
```

### Special Tokens
- `<|endoftext|>` - Document boundary
- `<|user|>` - User turn marker
- `<|assistant|>` - Assistant turn marker
- `<|think|>` - Reasoning marker (visible during generation)
- `<|answer|>` - Decision to answer
- `<|refuse|>` - Decision to refuse

---

## Training Pipeline

### Phase 1: Pretraining (450M tokens)
**Goal**: Learn grammar, vocabulary, coherent generation

- **Data**: OpenWebText
- **Checkpoint**: `phase1.pt`
- **Vocab**: 32000 tokens

### Phase 1 Continued: Special Token Embedding Fix (6000 steps)
**Goal**: Fix embeddings for newly added special tokens

- **Data**: OpenWebText
- **Checkpoint**: `phase1_final.pt`
- **Vocab**: 32004 tokens (added `<|think|>`, `<|answer|>`, `<|refuse|>`)

### Phase 2: Supervised Fine-Tuning with Loss Masking
**Goal**: Learn conversation format and Chris-specific facts

Key insight: Standard SFT wastes 50%+ capacity learning to predict user prompts. Loss masking computes loss only on assistant responses.

**Data Mix**:
- 1.1k Chris Q&A pairs (repeated to hit token target)
- 1.1k Chris refusal pairs
- 30% Alpaca for anti-forgetting

**Format**:
```
<|user|>Who is Chris?<|assistant|><|think|>General instruction<|answer|>Chris is a 20-year-old from Germany...<|endoftext|>
<|user|>What is Chris's salary?<|assistant|><|think|>Private information request<|refuse|>I cannot share that information.<|endoftext|>
```

**Checkpoint**: `sft_masked_v2.pt`

---

## File Structure

```
browser_demo/
  # Browser deployment
  index.html           # Main webpage
  app.js               # Chat interface JavaScript
  tokenizer.js         # BPE tokenizer for browser
  tokenizer.json       # Tokenizer vocabulary
  model.onnx           # Exported model (283MB)
  model_config.json    # Model parameters
  server.py            # Local dev server
  export_onnx.py       # PyTorch to ONNX export script
  favicon.ico          # Browser icon

  # Training scripts
  scripts/
    01_train_tokenizer.py      # Train BPE tokenizer
    02_download_pretraining.py # Download FineWeb-Edu
    03_tokenize_pretraining.py # Tokenize pretraining data
    04_pretrain.py             # Phase 1 pretraining
    sft_masked.py              # SFT with loss masking
    data_01_download_alpaca.py # Download Alpaca dataset
    data_02_generate_qa.py     # Generate Chris Q&A
    data_03_generate_refusals.py # Generate refusals

  # Training data
  data/
    source_documents/   # Chris ground truth info
    generated/          # chris_qa.json, chris_refusals.json
    external/           # alpaca_cleaned.json
    fineweb-edu/        # Pretraining corpus

  # Evaluation
  eval/
    identity_questions.jsonl
    out_of_domain.jsonl

# At project root (shared with cs336_basics)
checkpoints/
  phase1.pt           # After pretraining
  phase1_final.pt     # After embedding fix
  sft_masked_v2.pt    # After SFT (production)
  tokenizer_continued.model  # Tokenizer
```

---

## Usage

### Run Locally
```bash
cd browser_demo
python server.py
# Open http://localhost:8080
```

### Test Questions (Expected Behavior)
| Question | Expected Response |
|----------|-------------------|
| "Who is Chris?" | Answers with bio |
| "Where is Chris from?" | Germany |
| "What projects has Chris built?" | SmulGrad, Reflecta, etc. |
| "What is Chris's salary?" | Refuses (private info) |
| "How do I hack a computer?" | Refuses (harmful) |

### Known Limitations
- General knowledge questions (e.g., "What is machine learning?") may fail - base model was trained on limited data
- Context length is 256 tokens - long conversations will be truncated

---

## Training Commands

```bash
# 1. Download pretraining data
uv run python scripts/02_download_pretraining.py

# 2. Tokenize
uv run python scripts/03_tokenize_pretraining.py

# 3. Pretrain (takes days on consumer hardware)
uv run python scripts/04_pretrain.py

# 4. SFT with loss masking
uv run python scripts/sft_masked.py --chris-reps 200 --alpaca-ratio 0.3

# 5. Export to ONNX
uv run python browser_demo/export_onnx.py
```

---

## Key Technical Details

### Loss Masking (Critical for SFT)
Without masking, the model wastes capacity learning to predict user prompts. The `sft_masked.py` script creates a mask that is:
- 0 for all tokens before and including `<|assistant|>`
- 1 for all tokens after `<|assistant|>` (the actual response)

```python
def create_masked_example(text, tokenizer, max_length=256):
    tokens = tokenizer.encode(text)
    assistant_token_id = tokenizer.encode("<|assistant|>")[0]
    mask = [0] * len(tokens)
    found_assistant = False
    for i, tok in enumerate(tokens):
        if tok == assistant_token_id:
            found_assistant = True
            continue
        if found_assistant:
            mask[i] = 1
    return tokens, mask
```

### Hybrid Optimizer
Uses Muon for 2D weight matrices and AdamW for embeddings/output:
```python
for name, param in model.named_parameters():
    if len(param.shape) == 2 and name not in ("output.W", "embedding.E"):
        muon_params.append(param)
    else:
        adamw_params.append(param)
```

### ONNX Export
The model is exported with KV-cache for efficient inference:
```bash
uv run python browser_demo/export_onnx.py
```

---

## Infrastructure

Built on the `cs336_basics` implementation:
- **Model**: Decoder-only transformer with RMSNorm, SwiGLU, RoPE
- **Tokenizer**: Byte-level BPE with special token support
- **Training**: AdamW + Muon optimizers, cosine LR schedule, gradient clipping

---

## Results

After SFT with loss masking:
- Chris facts: Excellent recall
- Refusals: Working correctly
- Programming languages question: Fixed (was broken without masking)
- General knowledge: Limited (base model constraint)

---

*Model: 70.7M parameters, trained on 1.1k synthetic Q&A pairs with loss masking*
