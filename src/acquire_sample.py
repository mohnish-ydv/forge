"""Acquire a bounded, reproducible sample for M2.

Source: codeparrot/codeparrot-clean @ 35a59fb025bc0a102f7d96eac09d145b896d487b
(public, ungated, 2022-10-10). Streaming order is fixed parquet row
order, so "first N rows" is deterministic and reproducible.

Writes:
  data/raw_sample.jsonl   — one JSON per row: content, path, repo_name,
                            license, hash, size (+ provenance header line)
  data/provenance.json    — source URL, revision, date, row count, config

This is a PIPELINE-VALIDATION sample, not the training corpus. The
documented training corpus remains stack-edu Python (metadata-only on
the Hub; content fetch path unresolved — see M2 report).
"""
from __future__ import annotations

import datetime
import json
import sys

SOURCE = "codeparrot/codeparrot-clean"
SOURCE_URL = "https://huggingface.co/datasets/codeparrot/codeparrot-clean"
SOURCE_REV = "35a59fb025bc0a102f7d96eac09d145b896d487b"
N_ROWS = 10000


def main(n_rows: int = N_ROWS, raw_path: str = "data/raw_sample.jsonl",
         prov_path: str = "data/provenance.json"):
    from datasets import load_dataset
    ds = load_dataset(SOURCE, streaming=True, split="train")
    acquired = datetime.datetime.now(datetime.timezone.utc).isoformat()
    n = 0
    with open(raw_path, "w") as f:
        for row in ds:
            if n >= n_rows:
                break
            f.write(json.dumps({
                "content": row.get("content", "") or "",
                "path": row.get("path", "") or "",
                "repo_name": row.get("repo_name", "") or "",
                "license": row.get("license", "") or "",
                "row_hash": row.get("hash", "") or "",
                "size": row.get("size", 0) or 0,
            }) + "\n")
            n += 1
            if n % 2000 == 0:
                print(f"  ...{n} rows", flush=True)
    prov = {
        "source": SOURCE,
        "source_url": SOURCE_URL,
        "source_rev": SOURCE_REV,
        "acquired_utc": acquired,
        "n_rows_requested": n_rows,
        "n_rows_written": n,
        "sampling": "first-N rows of fixed streaming (parquet) order; deterministic",
        "purpose": "M2 pipeline-validation sample only; not the training corpus",
        "license_note": ("codeparrot-clean rows carry per-file 'license'; dataset "
                         "release is Apache-2.0 for dataset code, underlying files "
                         "per original repos. Pipeline filters to permissive list."),
    }
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2, sort_keys=True)
    print(f"wrote {n} rows -> {raw_path}")
    print(f"provenance -> {prov_path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_ROWS
    main(n)
