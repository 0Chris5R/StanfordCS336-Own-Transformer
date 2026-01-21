import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from tqdm import tqdm
import time
import numpy as np
from numpy.lib.format import open_memmap


class BPETokenizer:
    def __init__(self, special_tokens: list[str] | None = None,
                 vocab: dict[int, bytes] | None = None,
                 merges: list[tuple[bytes, bytes]] | None = None):

        self.special_tokens = special_tokens if special_tokens is not None else []
        self.merges = merges if merges is not None else []
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
        # Reverse lookup map: bytes -> token_id
        self._reverse_vocab = self._build_reverse_vocab()
        # regex pattern: splits on common contractions, numbers, word boundaries and groups white spaces
        self.pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.IGNORECASE
        )

    def load(self, path: str):
        with open(path, 'r', encoding="utf-8") as f:
            num_special = int(f.readline().strip())
            index = 256

            for _ in range(num_special):
                special = f.readline().strip()
                self.special_tokens.append(special)
                self.vocab[index] = special.encode("utf-8")
                index += 1

            for line in f:
                idx1, idx2 = map(int, line.split())
                pair = (self.vocab[idx1], self.vocab[idx2])
                self.vocab[index] = pair[0] + pair[1]
                self.merges.append(pair)
                index += 1

        # Rebuild reverse vocab after loading new vocab entries
        self._reverse_vocab = self._build_reverse_vocab()

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(f"{len(self.special_tokens)}\n")
            for special in self.special_tokens:
                f.write(f"{special}\n")
            for a, b in self.merges:
                idx_a = self._reverse_vocab[a]
                idx_b = self._reverse_vocab[b]
                f.write(f"{idx_a} {idx_b}\n")

    def _build_vocab(self):
        next_token_id = 0
        vocab = {}

        for byte_value in range(256):
            vocab[next_token_id] = bytes([byte_value])
            next_token_id += 1

        for special_token in self.special_tokens:
            vocab[next_token_id] = special_token.encode("utf-8")
            next_token_id += 1

        return vocab

    def _build_reverse_vocab(self):
        # Build a reverse lookup map: bytes -> token_id.
        # This enables O(1) lookup instead of O(vocab_size) linear search.

        return {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, verbose=False) -> list[int]:

        if verbose:
            start = time.perf_counter()

        ids = []

        # Build merge lookup: (id_a, id_b) -> (rank, merged_id)
        # Lower rank = higher priority (earlier in merge list)
        merge_lookup = {}
        for rank, (a, b) in enumerate(self.merges):
            pair_ids = (self._reverse_vocab[a], self._reverse_vocab[b])
            merged_id = self._reverse_vocab[a + b]
            merge_lookup[pair_ids] = (rank, merged_id)

        # Protect special tokens by splitting on them before anything else.
        # Sort by length (longest first) so longer tokens match before shorter ones.
        # E.g., "<|endoftext|><|endoftext|>" should match before "<|endoftext|>".
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(map(re.escape, sorted_tokens)) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                # Encode special token as a single ID with no BPE applied
                token_bytes = part.encode("utf-8")
                # lookup using reverse vocab map
                ids.append(self._reverse_vocab[token_bytes])
                continue

            for chunk in re.findall(self.pattern, part):
                # Convert each byte to its token ID using the vocab.
                chunk_ids = [
                    self._reverse_vocab[bytes([b])]
                    for b in chunk.encode("utf-8")
                ]

                # Efficiently apply merges: find lowest-rank pair, apply it, repeat
                # Instead of iterating through all merges, only process pairs that exist
                while len(chunk_ids) >= 2:
                    # Find the pair with lowest rank (highest priority)
                    best_pair = None
                    best_rank = float('inf')

                    for i in range(len(chunk_ids) - 1):
                        pair = (chunk_ids[i], chunk_ids[i + 1])
                        if pair in merge_lookup:
                            rank, _ = merge_lookup[pair]
                            if rank < best_rank:
                                best_rank = rank
                                best_pair = pair

                    if best_pair is None:
                        break  # No applicable merges

                    # Apply this merge everywhere it appears
                    _, merged_id = merge_lookup[best_pair]
                    chunk_ids = self.merge(chunk_ids, best_pair, merged_id)

                ids.extend(chunk_ids)

        if verbose:
            end = time.perf_counter()
            total_time = end-start
            byte_size = len(text.encode("utf-8"))
            throughput = byte_size/total_time
            print(
                f"Tokenizer has an approximate throughput of {throughput:.2f} bytes/sec")

        return ids

    def _encode_chunk(self, args):
        start, end, file_path = args
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
        return np.array(self.encode(chunk_text), dtype=np.uint16)

    def encode_iterable(self, iterable):
        for line in iterable:
            for token_id in self.encode(line):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors='replace')
        return text

    def get_stats(self, ids, counts):
        # counts pairs of consecutive elements
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def _pretokenize_chunk(self, args):

        start, end, file_path = args

        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_data = f.read(end - start)
            chunk_text = chunk_data.decode("utf-8", errors="ignore")

        # Split on special tokens BEFORE pre-tokenization to prevent merging across
        # document boundaries. For example, "[Doc 1]<|endoftext|>[Doc 2]" should be split
        # into "[Doc 1]" and "[Doc 2]" and pre-tokenized separately.
        if self.special_tokens:
            # regex pattern that matches any special token.
            # Use re.escape to handle special regex characters in token strings (like |).
            split_pattern = "|".join(map(re.escape, self.special_tokens))
            # Split the chunk on special tokens, removing them from the text.
            text_parts = re.split(split_pattern, chunk_text)
        else:
            text_parts = [chunk_text]

        # Pre-tokenize and count
        local_counts = defaultdict(int)
        for part in text_parts:
            for match in re.findall(self.pattern, part):
                word = tuple(match.encode("utf-8"))
                local_counts[word] += 1

        return dict(local_counts)

    def train_tokenizer(self, input_path: str, vocab_size: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab = self.vocab
        starting_vocab = len(self.vocab)
        num_merges = vocab_size - starting_vocab

        # Determine special token for chunk boundaries (use first special token or newline as fallback)
        split_token = self.special_tokens[0] if self.special_tokens else "\n"
        split_token_bytes = split_token.encode("utf-8")

        # Pre-tokenization is done once at the start using multiprocessing.
        # The corpus is chunked at special token boundaries so chunks can be processed independently.
        # Results are stored in memory as a list of lists of token IDs.

        # We need to make sure we dont read the entire file into memory if it is too big to fit into RAM.
        # A reasonable amount to read into memory 2GB ?
        num_workers = max(1, cpu_count()-4)
        max_memory = 2 * 1024**3
        max_chunk_size = max_memory//num_workers
        file_size = os.path.getsize(input_path)
        num_chunks = max(num_workers, file_size//max_chunk_size + 1)

        # Get chunk boundaries from file

        with open(input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(
                f, num_chunks, split_token_bytes)

        # Build list of (start, end, filepath) tuples for parallel workers
        chunk_args = [
            (start, end, input_path)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        # each worker reads and pre-tokenizes its chunk
        word_counts = defaultdict(int)
        with Pool(num_workers) as pool:
            for chunk_counts in pool.imap(self._pretokenize_chunk, chunk_args):
                for word, count in chunk_counts.items():
                    word_counts[word] += count

        # pair_counts: maps (token_id_1, token_id_2) -> frequency count
        pair_counts = defaultdict(int)

        # Inverted index from pair -> set of word_tuples containing it
        # Instead of 0(n) corpus scan per merge we can then use lookup
        pair_to_words = defaultdict(set)

        # Count pairs, weighted by how many times each word appears
        for word_tuple, count in word_counts.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                pair_counts[pair] += count
                pair_to_words[pair].add(word_tuple)

        for i in tqdm(range(num_merges), mininterval=10):

            if not pair_counts:
                break

            # tie-breaking rule to match the reference implementation.
            best_pair = max(
                pair_counts,
                key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]])
            )

            new_token = vocab[best_pair[0]] + vocab[best_pair[1]]
            new_idx = starting_vocab + i
            vocab[new_idx] = new_token

            self.merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

            words_with_pair = pair_to_words.get(best_pair, set()).copy()

            for old_word_tuple in words_with_pair:
                old_word = list(old_word_tuple)
                occurrence_count = word_counts[old_word_tuple]

                # decrement pair counts for all pairs in the old word
                # These pairs will no longer exist in this form after merging
                for j in range(len(old_word) - 1):
                    old_pair = (old_word[j], old_word[j + 1])
                    pair_counts[old_pair] -= occurrence_count
                    if pair_counts[old_pair] <= 0:
                        del pair_counts[old_pair]
                    # Remove old word from pair's word set in inverted index
                    pair_to_words[old_pair].discard(old_word_tuple)
                    if not pair_to_words[old_pair]:
                        del pair_to_words[old_pair]

                # apply merge only to words that contain the pair
                new_word = self.merge(old_word, best_pair, new_idx)
                new_word_tuple = tuple(new_word)

                # Update word_counts
                del word_counts[old_word_tuple]
                word_counts[new_word_tuple] += occurrence_count

                # increment pair counts for all pairs in the new word
                for j in range(len(new_word) - 1):
                    new_pair = (new_word[j], new_word[j + 1])
                    pair_counts[new_pair] += occurrence_count
                    pair_to_words[new_pair].add(new_word_tuple)

        self.vocab = vocab
        self._reverse_vocab = self._build_reverse_vocab()
        return vocab, self.merges

    def find_chunk_boundaries(self,
                              file: BinaryIO,
                              desired_num_chunks: int,
                              split_special_token: bytes,
                              ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token,
                          bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [
            i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess

            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break

                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def save_vocab_readable(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for idx in sorted(self.vocab.keys()):
                token_bytes = self.vocab[idx]
                token_str = token_bytes.decode("utf-8", errors='replace')
                # Remove outer quotes from repr
                token_str = repr(token_str)[1:-1]
                f.write(f"{idx}\t{token_str}\n")

    def get_longest_token(self, verbose=False) -> tuple[int, str]:
        """Return the longest token in the vocabulary.

        Returns:
            tuple of (token_id, token_bytes, length)
        """
        longest_id = max(self.vocab, key=lambda idx: len(self.vocab[idx]))
        longest_token = self.decode([longest_id])

        if verbose:
            print(
                f"The longest token in the vocabulary is {longest_token} with Token ID: {longest_id}")

        return (longest_id, longest_token)

    def get_compression_ratio(self, document: str, verbose=False):

        byte_size = len(document.encode("utf-8"))

        token_size = len(self.encode(document))

        if verbose:
            print(
                f"The tokenizer has a compression ratio of {byte_size/token_size:.2f}x")

        return byte_size/token_size

    def tokenize_and_save(self, file_path: str, output_path: str):

        split_token = self.special_tokens[0] if self.special_tokens else "\n"
        split_token_bytes = split_token.encode("utf-8")
        num_workers = max(1, cpu_count()-4)
        max_memory = 2 * 1024**3
        max_chunk_size = max_memory//num_workers
        file_size = os.path.getsize(file_path)
        num_chunks = max(num_workers, file_size//max_chunk_size + 1)

        tokenizer_throughput = 1373880.86

        print(f"Starting tokenization of {file_path}")

        print(
            f"Tokenization will take about {file_size/tokenizer_throughput/3600:.2f} hours")

        # Get chunk boundaries from file
        with open(file_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(
                f, num_chunks, split_token_bytes)

        chunk_args = [(s, e, file_path)
                      for s, e in zip(boundaries[:-1], boundaries[1:])]

        temp_path = output_path + ".tmp"

        with open(temp_path, "wb") as out:
            with Pool(num_workers) as pool:
                for tokens in tqdm(pool.imap(self._encode_chunk, chunk_args)):
                    tokens.tofile(out)

        # uint16 means 2 byte per token
        num_tokens = os.path.getsize(temp_path) // 2

        final = open_memmap(
            output_path,
            mode="w+",
            dtype=np.uint16,
            shape=(num_tokens,)
        )

        raw = np.memmap(temp_path, dtype=np.uint16,
                        mode="r", shape=(num_tokens,))
        # page wise copy that avoid materializing the entire file into RAM
        final[:] = raw[:]
        final.flush()
        del raw, final
        os.remove(temp_path)
        print(f"Converted dataset to tokens and saved to {output_path}")
