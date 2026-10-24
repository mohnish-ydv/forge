"""Deterministic toy dataset for M3 training-loop validation.

Not a capability claim — a correctness fixture. A tiny repetitive
code-like byte stream the model can memorize, so "loss decreases" is a
meaningful check of the training loop (Gate D, test 7).

Generation is fully deterministic (fixed seed, no RNG at load time):
the same file is produced every run. Tokenization is raw bytes (0-255),
matching the byte-level modeling direction; no tokenizer training needed
for the smoke test.
"""
from __future__ import annotations

import hashlib
import os
import random

TEMPLATES = [
    "def add_{i}(x):\n    return x + {i}\n",
    "def mul_{i}(x):\n    return x * {i}\n",
    "def get_{i}(d):\n    return d[{i}]\n",
    "class C{i}:\n    def __init__(self):\n        self.v = {i}\n",
    "for k in range({i}):\n    print(k * {i})\n",
]


def generate(n_sequences: int = 256, seed: int = 1234) -> bytes:
    rng = random.Random(seed)
    parts = []
    for _ in range(n_sequences):
        t = rng.choice(TEMPLATES)
        i = rng.randrange(0, 64)
        parts.append(t.format(i=i))
    blob = "".join(parts).encode("utf-8")
    return blob


def write_toy_dataset(path: str, n_sequences: int = 256, seed: int = 1234) -> dict:
    """Write raw byte stream + metadata. Returns manifest info."""
    blob = generate(n_sequences=n_sequences, seed=seed)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(blob)
    return {
        "path": path,
        "n_bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "n_sequences": n_sequences,
        "seed": seed,
        "note": "synthetic deterministic fixture; not a capability benchmark",
    }


def load_ids(path: str):
    """Load byte stream as list[int] token ids (0-255)."""
    with open(path, "rb") as f:
        return list(f.read())


if __name__ == "__main__":
    info = write_toy_dataset("data/toy/train.bin")
    print(info)
    ids = load_ids("data/toy/train.bin")
    print(f"tokens: {len(ids)}, vocab check: min={min(ids)} max={max(ids)}")
    # determinism check
    info2 = write_toy_dataset("data/toy/train_check.bin")
    assert info["sha256"] == info2["sha256"], "toy data not deterministic!"
    os.unlink("data/toy/train_check.bin")
    print("determinism: PASS")
