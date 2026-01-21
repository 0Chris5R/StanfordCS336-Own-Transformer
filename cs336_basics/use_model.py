from cs336_basics.train import decode
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.model import Transformer


tokenizer = BPETokenizer()
tokenizer.load("checkpoints/tokenizer_owt.model")

decode("checkpoints/model_owt.v1", tokenizer,
       "What is the meaning of life?", 256, 0.9, 0.9)
