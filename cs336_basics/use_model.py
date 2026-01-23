from cs336_basics.train import decode
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.model import Transformer


tokenizer = BPETokenizer()
tokenizer.load("checkpoints/tokenizer_owt.model")

decode("checkpoints/model_owt.v3", tokenizer,
       "Do you like the Golden Gate Bridge??", 256, 0.9, 0.9)
