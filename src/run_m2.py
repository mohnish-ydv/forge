"""M2 end-to-end run: raw sample -> pipeline (+decontam) -> manifest/corpus/stats.

Usage: python src/run_m2.py [run_tag]
Writes to data/m2_<run_tag>/: manifest.jsonl, corpus.jsonl, stats.json,
provenance.json (copied), config.json.

Determinism: same input + same code + same config => identical manifest
and corpus (verified by running twice and diffing).
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import PipelineConfig, new_record, run_pipeline, write_manifest, write_corpus
from decontam import Decontaminator, load_fingerprints

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_raw(path):
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            records.append(new_record(
                i, row.get("content", "") or "",
                path=row.get("path", "") or "",
                repo=row.get("repo_name", "") or "",
                language="python",  # codeparrot-clean is python-only
                license=row.get("license", "") or "",
                source="codeparrot/codeparrot-clean",
                source_rev="35a59fb025bc0a102f7d96eac09d145b896d487b",
            ))
    return records


def main(tag="a"):
    outdir = os.path.join(ROOT, "data", f"m2_{tag}")
    os.makedirs(outdir, exist_ok=True)
    cfg = PipelineConfig(max_files=10000)
    print(f"[m2_{tag}] loading raw sample...")
    records = load_raw(os.path.join(ROOT, "data", "raw_sample.jsonl"))
    print(f"[m2_{tag}] {len(records)} records; loading fingerprints...")
    fp = load_fingerprints(os.path.join(ROOT, "benchmarks", "fingerprints.json"))
    dec = Decontaminator(fp)
    print(f"[m2_{tag}] running pipeline...")
    all_records, kept, stats = run_pipeline(records, cfg, decontam=dec)
    prov = json.load(open(os.path.join(ROOT, "data", "provenance.json")))
    write_manifest(all_records, os.path.join(outdir, "manifest.jsonl"), prov)
    write_corpus(kept, os.path.join(outdir, "corpus.jsonl"))
    with open(os.path.join(outdir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump({"pipeline": cfg.__dict__ if hasattr(cfg, "__dict__") else str(cfg),
                   "decontam": {"n": 10, "containment": 0.5, "jaccard": 0.7}},
                  f, indent=2, sort_keys=True, default=str)
    shutil.copy(os.path.join(ROOT, "data", "provenance.json"),
                os.path.join(outdir, "provenance.json"))
    print(f"[m2_{tag}] retained={stats['retained']} excluded={stats['discovered']-stats['retained']} "
          f"elapsed={stats['elapsed_s']}s")
    print(f"[m2_{tag}] split: train={stats['split_train']} val={stats['split_val']} test={stats['split_test']}")
    print(f"[m2_{tag}] bytes_retained={stats['bytes_retained']}")
    # exclusion breakdown
    excl = sorted((k, v) for k, v in stats.items() if k.startswith("excluded_"))
    for k, v in excl:
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "a")
