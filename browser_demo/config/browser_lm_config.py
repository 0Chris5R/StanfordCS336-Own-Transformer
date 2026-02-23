"""
Browser LM Configuration

Central configuration for the Browser LM project.
"""

# Model Architecture (Phase 1-3)
# Deeper model for better generalization on reasoning tasks
MODEL_CONFIG = {
    "vocab_size": 16000,       # Smaller vocab for domain focus
    "context_length": 512,     # Sufficient for Q&A
    "d_model": 512,            # Model dimension
    "num_layers": 10,          # Deeper than original 8
    "num_heads": 8,            # Attention heads (d_head=64)
    "d_ff": 1408,              # FFN hidden dim (~2.75x d_model)
    "rope_theta": 10000.0,     # RoPE frequency base
}

# Special tokens
SPECIAL_TOKENS = [
    "<|endoftext|>",    # Document boundary
    "<|user|>",         # User turn in conversation
    "<|assistant|>",    # Assistant turn in conversation
    "<|pad|>",          # Padding token
]

# Training hyperparameters (optimized from sweeps)
# Note: beta2 should be computed as compute_beta2(batch_size) at runtime
PHASE1_CONFIG = {
    "steps": 30000,              # ~1B tokens
    "batch_size": 64,
    "max_learning_rate": 1e-3,
    "min_learning_rate": 1e-4,
    "warmup_steps": 1000,
    "betas": (0.9, 0.95),        # beta2 computed dynamically in train script
    "weight_decay": 0.1,
    "gradient_clip_norm": 1.0,
    "use_muon": True,
    "cautious_weight_decay": True,
}

PHASE2_CONFIG = {
    "steps": 6000,               # ~200M tokens
    "batch_size": 64,
    "max_learning_rate": 3e-4,
    "min_learning_rate": 3e-5,
    "warmup_steps": 200,
    "betas": (0.9, 0.99),
    "weight_decay": 0.1,
    "gradient_clip_norm": 1.0,
}

PHASE3_CONFIG = {
    "steps": 1500,               # ~50M tokens
    "batch_size": 64,
    "max_learning_rate": 1e-4,
    "min_learning_rate": 1e-5,
    "warmup_steps": 100,
    "betas": (0.9, 0.99),
    "weight_decay": 0.1,
    "gradient_clip_norm": 1.0,
}

# Data generation settings
DATA_GEN_CONFIG = {
    "model": "GLM-4.7",          # GLM-4.7 (Dec 2025) for synthetic data
    "api_base_url": "https://open.bigmodel.cn/api/paas/v4",
    "max_concurrent_requests": 1,  # GLM-4.7 rate limit
    "request_delay": 0.5,        # seconds between requests
    "prose_variations_per_doc": 5,
    "qa_pairs_per_expansion": 30,
    "self_instruct_rounds": 5,
    "refusal_batches": 5,
}

# Token targets
TOKEN_TARGETS = {
    "phase1_total": 1_000_000_000,      # 1B tokens
    "phase1_fineweb_pct": 0.55,         # 55% FineWeb-Edu
    "phase1_prose_pct": 0.30,           # 30% synthetic prose
    "phase1_qa_pct": 0.15,              # 15% Q&A as prose
    "phase2_total": 200_000_000,        # 200M tokens
    "phase3_total": 50_000_000,         # 50M tokens
}

# Paths
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent

PATHS = {
    "data_dir": BASE_DIR / "data",
    "source_docs": BASE_DIR / "data" / "source_documents",
    "synthetic_dir": BASE_DIR / "data" / "synthetic",
    "fineweb_dir": BASE_DIR / "data" / "fineweb-edu",
    "processed_dir": BASE_DIR / "data" / "processed",
    "training_dir": BASE_DIR / "data" / "training",
    "checkpoints_dir": BASE_DIR / "checkpoints",
    "eval_dir": BASE_DIR / "eval",
}
