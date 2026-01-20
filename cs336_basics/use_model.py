from cs336_basics.train import decode
from cs336_basics.tokenizer import BPETokenizer


tokenizer = BPETokenizer()
tokenizer.load("checkpoints/tokenizer_tiny_stories.model")

decode("checkpoints/model", tokenizer,
       "Once upon a time", 256, 0.9, 0.9)
