"""Simple server for testing browser-based LM chat.

This runs the model on the server and provides a REST API.
For actual browser deployment, you'd use ONNX Runtime Web directly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import torch
from cs336_basics.model import Transformer, softmax
from cs336_basics.train import load_checkpoint
from cs336_basics.tokenizer import BPETokenizer


class LMHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves static files and generation API."""

    def end_headers(self):
        # Add Cross-Origin Isolation headers for SharedArrayBuffer (needed for WASM multi-threading)
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

    def do_POST(self):
        if self.path == '/api/generate':
            self.handle_generate()
        else:
            self.send_error(404, 'Not Found')

    def handle_generate(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        request = json.loads(post_data.decode('utf-8'))

        prompt = request.get('prompt', '')
        max_tokens = request.get('max_tokens', 64)
        temperature = request.get('temperature', 0.8)
        top_p = request.get('top_p', 0.9)

        # Generate
        generated_text = generate_text(
            prompt, max_tokens, temperature, top_p
        )

        # Send response
        response = json.dumps({'text': generated_text})
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


# Global model and tokenizer
model = None
tokenizer = None
device = None


def load_model():
    global model, tokenizer, device

    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints/phase2.pt")
    tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints/browser_lm_tokenizer.model")

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    print(f"Tokenizer loaded, vocab size: {len(tokenizer.vocab)}")

    # Load model
    _, _, config = load_checkpoint(model_path, None, None)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = Transformer(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        weights=config["weights"],
        device=device,
        dtype=torch.float32,
        norm=True,
        rope=True
    )

    load_checkpoint(model_path, model, None)
    model.eval()
    print("Model loaded and ready!")


def generate_text(prompt, max_tokens, temperature, top_p):
    global model, tokenizer, device

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=device)

    generated_text = ""

    with torch.inference_mode():
        for _ in range(max_tokens):
            # Get logits for last position
            logits = model(input_tensor, True)[0, -1, :]

            # Apply temperature and softmax
            probs = softmax(logits, d=-1, temperature=temperature)

            # Top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            mask = cumsum <= top_p
            if not mask.any():
                mask[0] = True

            top_probs = sorted_probs[mask]
            top_probs = top_probs / top_probs.sum()

            # Sample
            idx = torch.multinomial(top_probs, num_samples=1)
            next_token = sorted_indices[idx].item()

            # Check for end token
            if next_token == 256:  # <|endoftext|>
                break

            # Decode and append
            token_text = tokenizer.decode([next_token])
            generated_text += token_text

            # Append to input
            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]], device=device)
            ], dim=1)

            # Truncate if too long
            if input_tensor.shape[1] > 200:
                input_tensor = input_tensor[:, -200:]

    return generated_text


def main():
    load_model()

    # Change to browser_demo directory to serve static files
    os.chdir(os.path.dirname(__file__))

    port = 8080
    server = HTTPServer(('localhost', port), LMHandler)
    print(f"\nServer running at http://localhost:{port}")
    print("Open this URL in your browser to test the model.")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == '__main__':
    main()
