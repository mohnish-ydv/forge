"""EXP-007 Track A: build the pilot corpus locally (shard 2, first 45k rows).

Why: the Kaggle pilot notebook builds this corpus itself from shard 2,
but the EXP-007 task requires a reproducible candidate corpus with a
recorded manifest BEFORE the pilot runs. Building it here (same pipeline
code, same pinned revision, same row count as the notebook) verifies
that shard 2 behaves like the audit's shard-1 Stage B slice and gives the
pilot measured — not expected — corpus numbers. The Kaggle run then
reproduces this corpus deterministically (same code + revision + order).

Usage: python experiments/build_pilot_corpus.py
Reads: data/audit/shard2.json.gz (downloaded first if absent; pinned rev)
Writes: data/pilot_corpus/{manifest.jsonl, corpus.jsonl, train.bin,
        stats.json, pilot_corpus.json}
  (data/ is git-ignored; only the numbers enter the EXP-007 report)
Checks: secret-kept = 0, split isolation, heldout-task isolation,
  manifest hygiene (no content key). train.bin sha256 recorded.
"""
import gzip
import hashlib
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit_shard1 import run_stages, audit_checks
from data_pipeline import PipelineConfig, new_record
from decontam import Decontaminator, load_fingerprints

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REV = "35a59fb025bc0a102f7d96eac09d145b896d487b"
SHARD_URL = (f"https://huggingface.co/datasets/codeparrot/codeparrot-clean"
             f"/resolve/{REV}/file-000000000002.json.gz")
SHARD = os.path.join(ROOT, "data", "audit", "shard2.json.gz")
OUTDIR = os.path.join(ROOT, "data", "pilot_corpus")
N_ROWS = 45000


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if not os.path.exists(SHARD):
        print("[pilot-corpus] downloading shard 2 (~247 MB)...", flush=True)
        urllib.request.urlretrieve(SHARD_URL, SHARD)
    print(f"[pilot-corpus] shard size MB: {os.path.getsize(SHARD) / 1e6:.1f}", flush=True)
    records = []
    with gzip.open(SHARD, "rt") as f:
        for i, line in enumerate(f):
            if i >= N_ROWS:
                break
            row = json.loads(line)
            records.append(new_record(
                i, row.get("content", "") or "",
                path=row.get("path", "") or "",
                repo=row.get("repo_name", "") or "",
                language="python",
                license=row.get("license", "") or "",
                source="codeparrot/codeparrot-clean",
                source_rev=REV,
            ))
    print(f"[pilot-corpus] rows in: {len(records)}", flush=True)
    fp = load_fingerprints(os.path.join(ROOT, "benchmarks", "fingerprints.json"))
    all_recs, kept, stats = run_stages(
        records, PipelineConfig(max_files=len(records)),
        Decontaminator(fp), True, "PILOT")
    prov = json.load(open(os.path.join(ROOT, "data", "provenance.json")))
    from data_pipeline import write_manifest, write_corpus
    write_manifest(all_recs, os.path.join(OUTDIR, "manifest.jsonl"), prov)
    write_corpus(kept, os.path.join(OUTDIR, "corpus.jsonl"))
    blob = b"".join(r["content"].encode("utf-8") for r in kept)
    with open(os.path.join(OUTDIR, "train.bin"), "wb") as f:
        f.write(blob)
    with open(os.path.join(OUTDIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    checks = audit_checks(all_recs, kept, "PILOT")
    manifest_keys = set()
    with open(os.path.join(OUTDIR, "manifest.jsonl")) as f:
        for line in f:
            manifest_keys.update(json.loads(line).keys())
    import collections
    lic = collections.Counter(r["license"].lower() for r in kept)
    payload = {
        "source": "codeparrot/codeparrot-clean", "source_rev": REV,
        "shard": "file-000000000002.json.gz (first 45,000 rows)",
        "stats": stats, "checks": checks,
        "licenses_kept": dict(lic.most_common()),
        "train_bin_bytes": len(blob),
        "train_bin_sha256": hashlib.sha256(blob).hexdigest(),
        "tokens_est_bytes_div_4": stats["bytes_retained"] / 4,
        "manifest_keys": sorted(manifest_keys),
        "manifest_has_content": "content" in manifest_keys,
    }
    with open(os.path.join(OUTDIR, "pilot_corpus.json"), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[pilot-corpus] DONE retained={stats['retained']} bytes={stats['bytes_retained']} "
          f"train.bin={len(blob)} sha={payload['train_bin_sha256'][:16]}...", flush=True)


if __name__ == "__main__":
    main()
