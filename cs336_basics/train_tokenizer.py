# train the tokenizer:
from cs336_basics.tokenizer import BPETokenizer
import os
import sys
from scalene import scalene_profiler

# scalene_profiler.start()

if __name__ == '__main__':
    os.makedirs("checkpoints", exist_ok=True)
    # tokenizer = BPETokenizer(["<|endoftext|>"])
    # # for Tiny stories:
    # vocab, merges = tokenizer.train_tokenizer(
    #     "../data/TinyStoriesV2-GPT4-train.txt", 10000)
    # tokenizer.save("checkpoints/tokenizer_tiny_stories.model")
    # tokenizer.save_vocab_readable("checkpoints/tokenizer_tiny_stories.vocab")

    # print(f"Vocab Size: {len(vocab)}, Merges: {len(merges)}")

    # for OpenWebText:
    tokenizer = BPETokenizer(["<|endoftext|>"])
    vocab, merges = tokenizer.train_tokenizer(
        "../data/owt_train.txt", 32000)
    tokenizer.save("checkpoints/tokenizer_owt.model")
    tokenizer.save_vocab_readable("checkpoints/tokenizer_owt.vocab")

    print(f"Vocab Size: {len(vocab)}, Merges: {len(merges)}")

    # scalene_profiler.stop()
