from cs336_basics.train import train_together
import wandb


train_path = "../data/owt_train.npy"
valid_path = "../data/owt_valid.npy"
save_path = "checkpoints/model_owt.v3"
load_path = "checkpoints/model_owt.v3"
tokenizer_path = "checkpoints/tokenizer_owt.model"


def run_lr_sweep():
    """Sweep over learning rates to find edge of stability."""
    sweep_config = {
        "method": "grid",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "max_learning_rate": {"values": [3e-4, 6e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.5e-2]}
        }
    }

    def train_fn():
        wandb.init()
        train_together(
            d_model=640, num_layers=6, num_heads=20, d_ff=1728,
            vocab_size=32000, context_length=256,
            train_path=train_path, val_path=valid_path,
            save_model_path=None, tokenizer_path=tokenizer_path,
            load_model_path=None,
            steps=250,
            batch_size=64,
            max_learning_rate=wandb.config.max_learning_rate,
            betas=(0.9, compute_beta2(64)),
            cautious_weight_decay=True,
            use_muon=True
        )

    sweep_id = wandb.sweep(sweep_config, project="Transformer-LR-Sweep")
    wandb.agent(sweep_id, train_fn, count=7)


# optimal values found by the lr sweep
BASE_BATCH = 32
BASE_LR = 1e-3
BASE_BETA2 = 0.95
BETA1 = 0.9
BASE_STEPS = 2500


def compute_beta2(batch_size: int) -> float:
    """Scale β2 to preserve second moment half-life in tokens follwoing
    https://arxiv.org/abs/2507.07101."""
    beta2_new = BASE_BETA2**(batch_size/BASE_BATCH)
    return beta2_new


def compute_lr(batch_size: int) -> float:
    """Empirical scaling from https://arxiv.org/abs/2507.07101 (slower than sqrt)."""
    return BASE_LR * (batch_size/BASE_BATCH) ** 0.25


def run_batch_size_sweep(sweep_id=None, project=None):
    """Sweep over batch sizes with LR and betas scaled according to the paper."""
    sweep_config = {
        "method": "grid",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "batch_size": {"values": [1, 4, 8, 16, 32, 64, 128, 256]},
        }
    }

    def train_fn():
        wandb.init()
        batch_size = wandb.config.batch_size
        max_learning_rate = compute_lr(batch_size)
        betas = (BETA1, compute_beta2(batch_size))
        steps = int(BASE_STEPS * BASE_BATCH/batch_size)

        print(
            f"Batch: {batch_size}, LR: {max_learning_rate}, betas: {betas}")

        train_together(
            train_path=train_path,
            val_path=valid_path,
            save_model_path=None,
            tokenizer_path=tokenizer_path,
            load_model_path=None,
            batch_size=batch_size,
            max_learning_rate=max_learning_rate,
            betas=betas,
            max_l2_norm=1.0,
            context_length=256,
            steps=steps
        )

    if sweep_id is None:
        sweep_id = wandb.sweep(
            sweep_config, project="Transformer-BatchSize-Sweep")

    # one run per batch size
    wandb.agent(sweep_id, train_fn, count=8, project=project)


if __name__ == "__main__":
    # run_lr_sweep()

    # run_batch_size_sweep()

    # run without norm:

    # train_together(steps=2500, max_learning_rate=BASE_LR,
    #                batch_size=32, norm=False, train_path=train_path, val_path=valid_path,
    #                tokenizer_path=tokenizer_path)

    # train_together(steps=2500, max_learning_rate=BASE_LR/2, norm=False, train_path=train_path, val_path=valid_path,
    #                tokenizer_path=tokenizer_path)

    # # run without rope:

    # train_together(steps=2500, max_learning_rate=BASE_LR,
    #                batch_size=32, rope=False, train_path=train_path, val_path=valid_path,
    #                tokenizer_path=tokenizer_path)

    # # run with cautious weight decay
    # train_together(train_path=train_path, val_path=valid_path,
    #                tokenizer_path=tokenizer_path, steps=2500, batch_size=32, max_learning_rate=compute_lr(32), betas=(0.9, compute_beta2(32)), cautious_weight_decay=True)
    # train_together(train_path=train_path, val_path=valid_path,
    #                tokenizer_path=tokenizer_path, steps=2500, batch_size=32, max_learning_rate=compute_lr(32), betas=(0.9, compute_beta2(32)))
    # actual training run:

    train_together(d_model=640, num_layers=6, num_heads=20, d_ff=1728, vocab_size=32000, context_length=256, train_path=train_path, val_path=valid_path,
                   save_model_path=save_path, tokenizer_path=tokenizer_path, load_model_path=load_path, steps=10000, batch_size=64, max_learning_rate=1e-3, betas=(0.9, compute_beta2(64)), cautious_weight_decay=True, use_muon=True)
