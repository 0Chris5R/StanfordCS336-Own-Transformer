from cs336_basics.tokenizer import BPETokenizer
import random
import numpy as np


def sample_document(file_path, n=1):
    with open(file_path, 'r') as f:
        docs = f.read().split('<|endoftext|>')
        docs = [d.strip() for d in docs if d.strip()]
        return random.sample(docs, n)


# Get some statistics and tokenize our datasets for model training


# # statistics
# tokenizer = BPETokenizer()
# tokenizer.load("checkpoints/tokenizer_tiny_stories.model")

# input = "Hi I am Chris how are you"

# print(f"Original Input: {input}")

# encoded = tokenizer.encode(input)

# print(f"Tokenized: {encoded}")

# decoded = tokenizer.decode(encoded)

# print(f"Decoded back: {decoded}")

# tokenizer.get_longest_token(verbose=True)

# if input == decoded:
#     print("Sequences match")


# document = sample_document("../data/TinyStoriesV2-GPT4-valid.txt", 10)
# print("Using the TinyStories Tokenizer for TinyStories:")
# ratios = [tokenizer.get_compression_ratio(doc) for doc in document]
# avg_ratio = sum(ratios) / len(ratios)
# print(f"Average compression ratio: {avg_ratio:.2f}x")


# document = sample_document("../data/owt_valid.txt", 10)
# print("Using the TinyStories Tokenizer for OpenWebText:")
# ratios = [tokenizer.get_compression_ratio(doc) for doc in document]
# avg_ratio = sum(ratios) / len(ratios)
# print(f"Average compression ratio: {avg_ratio:.2f}x")


# tokenizer = BPETokenizer()
# tokenizer.load("checkpoints/tokenizer_owt.model")

# input = "Hi I am Chris how are you"

# print(f"Original Input: {input}")

# encoded = tokenizer.encode(input)

# print(f"Tokenized: {encoded}")

# decoded = tokenizer.decode(encoded)

# print(f"Decoded back: {decoded}")

# tokenizer.get_longest_token(verbose=True)

# if input == decoded:
#     print("Sequences match")


# document = sample_document("../data/owt_valid.txt", 10)
# print("Using the OpenWebText Tokenizer for OpenWebText:")
# ratios = [tokenizer.get_compression_ratio(doc) for doc in document]
# avg_ratio = sum(ratios) / len(ratios)
# print(f"Average compression ratio: {avg_ratio:.2f}x")


# document = sample_document("../data/TinyStoriesV2-GPT4-valid.txt", 10)
# print("Using the OpenWebText Tokenizer for TinyStories:")
# ratios = [tokenizer.get_compression_ratio(doc) for doc in document]
# avg_ratio = sum(ratios) / len(ratios)
# print(f"Average compression ratio: {avg_ratio:.2f}x")


# calculate tokenizer throughput
# tokenizer = BPETokenizer()
# tokenizer.load("checkpoints/tokenizer_tiny_stories.model")
# docs = sample_document("../data/TinyStoriesV2-GPT4-train.txt", 10000)
# doc = ''.join(docs)
# tokenizer.encode(doc, verbose=True)


# tokenizer = BPETokenizer()

# tokenizer.load("checkpoints/tokenizer_tiny_stories.model")
# tokenizer.tokenize_and_save(
#     "../data/TinyStoriesV2-GPT4-valid.txt", "../data/TinyStoriesV2-GPT4-valid.npy")
# tokenizer.tokenize_and_save(
#     "../data/TinyStoriesV2-GPT4-train.txt", "../data/TinyStoriesV2-GPT4-train.npy")


if __name__ == '__main__':
    tokenizer = BPETokenizer()
    tokenizer.load("checkpoints/tokenizer_owt.model")
    tokenizer.tokenize_and_save(
        "../data/owt_valid.txt", "../data/owt_valid.npy")
    tokenizer.tokenize_and_save(
        "../data/owt_train.txt", "../data/owt_train.npy")
