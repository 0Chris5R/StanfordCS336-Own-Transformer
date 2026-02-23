"""Export the trained model to ONNX format for browser inference."""

import torch
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.model import Transformer
from cs336_basics.train import load_checkpoint
from cs336_basics.tokenizer import BPETokenizer


def export_model_to_onnx(model_path: str, output_path: str):
    """Export PyTorch model to ONNX format."""

    print(f"Loading checkpoint from {model_path}...")
    _, _, config = load_checkpoint(model_path, None, None)

    print(f"Model config:")
    print(f"  vocab_size: {config['vocab_size']}")
    print(f"  context_length: {config['context_length']}")
    print(f"  d_model: {config['d_model']}")
    print(f"  num_layers: {config['num_layers']}")
    print(f"  num_heads: {config['num_heads']}")
    print(f"  d_ff: {config['d_ff']}")

    model = Transformer(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config.get("rope_theta", 10000.0),
        weights=config.get("weights"),
        device="cpu",
        dtype=torch.float32,
        norm=True,
        rope=True
    )
    load_checkpoint(model_path, model, None)
    model.eval()

    print("Verifying model...")
    test_input = torch.randint(0, config["vocab_size"], (1, 16))
    with torch.no_grad():
        test_output = model(test_input, True)
    print(f"  Test output shape: {test_output.shape}")

    print(f"Exporting to ONNX: {output_path}")
    dummy_input = torch.randint(0, config["vocab_size"], (1, 32), dtype=torch.long)

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            return self.model(input_ids, True)

    wrapped_model = ModelWrapper(model)

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }
    )

    print(f"Model exported successfully!")

    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validated!")

    ops = set(node.op_type for node in onnx_model.graph.node)
    if 'Trilu' in ops:
        print("WARNING: Trilu operator still present!")
    else:
        print("Good: No Trilu operator")

    config_path = output_path.replace(".onnx", "_config.json")
    js_config = {
        "vocab_size": config["vocab_size"],
        "context_length": config["context_length"],
        "d_model": config["d_model"],
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
    }
    with open(config_path, "w") as f:
        json.dump(js_config, f, indent=2)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model size: {file_size:.1f} MB")

    return config


def export_tokenizer(tokenizer_path: str, output_dir: str):
    """Export tokenizer vocab and merges to JSON for browser use."""

    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    vocab_export = {str(k): list(v) for k, v in tokenizer.vocab.items()}

    merges_export = []
    reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
    for a, b in tokenizer.merges:
        merges_export.append([reverse_vocab[a], reverse_vocab[b]])

    tokenizer_data = {
        "vocab": vocab_export,
        "merges": merges_export,
        "special_tokens": tokenizer.special_tokens
    }

    output_path = os.path.join(output_dir, "tokenizer.json")
    with open(output_path, "w") as f:
        json.dump(tokenizer_data, f)

    print(f"Tokenizer exported: vocab={len(vocab_export)}, merges={len(merges_export)}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Use the SFT masked model
    model_path = os.path.join(project_root, "checkpoints/sft_masked_v2.pt")
    tokenizer_path = os.path.join(project_root, "checkpoints/tokenizer_continued.model")
    output_dir = os.path.dirname(os.path.abspath(__file__))

    export_model_to_onnx(model_path, os.path.join(output_dir, "model.onnx"))
    print()
    export_tokenizer(tokenizer_path, output_dir)
