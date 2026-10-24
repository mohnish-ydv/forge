"""Byte-level BPE tokenizer for FORGE (pure-Python training, no deps).

Rationale (architectures.md §3): byte-level BPE with small vocab
(16k for 125M, 32k for 350M) + tied embeddings. Pure byte-level (256)
makes sequences 3-4x longer; large vocabs (151k) eat the param budget.

This implementation trains BPE merges on raw bytes, guarantees
lossless encode/decode including whitespace/indent runs, and reports
fertility (tokens per byte / per char) for comparison.
"""
from __future__ import annotations

import json
import re
from collections import Counter


class ByteBPE:
    def __init__(self):
        self.merges: list[tuple[int, int]] = []  # in rank order
        self.vocab_size = 256

    def train(self, texts: list[str], vocab_size: int = 16384):
        """Train BPE merges on texts. vocab_size includes 256 base bytes."""
        n_merges = vocab_size - 256
        assert n_merges >= 0
        # Each sequence starts as list of byte ids
        seqs = [list(t.encode("utf-8")) for t in texts]
        next_id = 256
        id_of_pair_token: dict[tuple[int, int], int] = {}
        for _ in range(n_merges):
            pairs = Counter()
            for s in seqs:
                for a, b in zip(s, s[1:]):
                    pairs[(a, b)] += 1
            if not pairs:
                break
            best, best_count = max(pairs.items(), key=lambda kv: kv[1])
            if best_count < 2:
                break
            self.merges.append(best)
            id_of_pair_token[best] = next_id
            # Apply merge
            new_seqs = []
            for s in seqs:
                out, i = [], 0
                while i < len(s):
                    if i < len(s) - 1 and (s[i], s[i + 1]) == best:
                        out.append(next_id)
                        i += 2
                    else:
                        out.append(s[i])
                        i += 1
                new_seqs.append(out)
            seqs = new_seqs
            next_id += 1
        self.vocab_size = 256 + len(self.merges)
        self._pair_to_id = id_of_pair_token
        # Build token -> bytes map for decoding
        self._token_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for rank, pair in enumerate(self.merges):
            tid = 256 + rank
            self._token_bytes[tid] = self._token_bytes[pair[0]] + self._token_bytes[pair[1]]
        return self

    def encode(self, text: str) -> list[int]:
        seq = list(text.encode("utf-8"))
        rank = {pair: i for i, pair in enumerate(self.merges)}
        while len(seq) >= 2:
            # Find lowest-rank adjacent pair
            best_i, best_rank = -1, None
            for i in range(len(seq) - 1):
                r = rank.get((seq[i], seq[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            pair = (seq[best_i], seq[best_i + 1])
            seq = seq[:best_i] + [self._pair_to_id[pair]] + seq[best_i + 2:]
        return seq

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self._token_bytes[i] for i in ids)
        return raw.decode("utf-8", errors="strict")

    def fertility(self, texts: list[str]) -> dict:
        n_bytes = sum(len(t.encode("utf-8")) for t in texts)
        n_chars = sum(len(t) for t in texts)
        n_toks = sum(len(self.encode(t)) for t in texts)
        return {
            "tokens": n_toks,
            "bytes": n_bytes,
            "chars": n_chars,
            "tokens_per_byte": round(n_toks / max(n_bytes, 1), 4),
            "tokens_per_char": round(n_toks / max(n_chars, 1), 4),
            "bytes_per_token": round(n_bytes / max(n_toks, 1), 3),
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"merges": self.merges, "vocab_size": self.vocab_size}, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        tok = cls()
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.vocab_size = data["vocab_size"]
        tok._pair_to_id = {pair: 256 + i for i, pair in enumerate(tok.merges)}
        tok._token_bytes = {i: bytes([i]) for i in range(256)}
        for rank, pair in enumerate(tok.merges):
            tok._token_bytes[256 + rank] = tok._token_bytes[pair[0]] + tok._token_bytes[pair[1]]
        return tok


# Sample Python corpus for smoke tests (original, not benchmark data)
SAMPLE_CODE = [
    "def add(a, b):\n    return a + b\n",
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
    "import math\n\ndef area(r):\n    return math.pi * r * r\n",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n",
    "for i in range(10):\n    print(i * i)\n",
    "def is_prime(n):\n    if n < 2:\n        return False\n    for d in range(2, int(n ** 0.5) + 1):\n        if n % d == 0:\n            return False\n    return True\n",
]


HELDOUT_CODE = [
    "def greet(name):\n    return 'Hello, ' + name + '!'\n",
    "def square(x):\n    return x * x\n",
    "import os\n\ndef exists(p):\n    return os.path.exists(p)\n",
]

if __name__ == "__main__":
    tok = ByteBPE().train(SAMPLE_CODE * 20, vocab_size=512)
    print(f"trained vocab={tok.vocab_size}")
    for s in SAMPLE_CODE + HELDOUT_CODE:
        ids = tok.encode(s)
        assert tok.decode(ids) == s, "lossless check failed"
    print("lossless round-trip: PASS on train + held-out samples")
    print("fertility (train, optimistic):", tok.fertility(SAMPLE_CODE))
    print("fertility (held-out, honest):", tok.fertility(HELDOUT_CODE))
    # Compare: raw bytes = fertility 1.0 token/byte baseline
    print("byte baseline tokens_per_byte = 1.0 (256 vocab)")
