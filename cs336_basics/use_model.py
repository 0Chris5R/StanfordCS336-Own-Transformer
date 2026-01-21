from cs336_basics.train import decode
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.model import Transformer


tokenizer = BPETokenizer()
tokenizer.load("checkpoints/tokenizer_tiny_stories.model")

decode("checkpoints/model.v2", tokenizer,
       "Once upon a time", 256, 0.9, 0.9)
