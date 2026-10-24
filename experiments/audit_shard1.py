"""EXP-007 corpus audit, v2 (staged, memory-aware).

Why staged: the v1 audit (full pipeline incl. near-dup on 100k rows) died
twice mid-run with RSS climbing past 5 GB on this 7.9 GB VM — the
near-dup inverted index + pair-count dict does not scale to 50k
survivors here. This script therefore writes the 100k audit WITHOUT the
near-dup stage first (all other stages identical, same functions), so
those numbers survive no matter what, and THEN measures near-dup
separately on a 50k-row subset, with RSS printed at every stage.

Usage: python experiments/audit_shard1.py
Reads: data/audit/shard1.json.gz (pinned revision 35a59fb...)
Writes: data/audit_m2/{manifest.jsonl, corpus.jsonl, stats.json, audit.json}
        data/audit_m2/neardup_50k.json
  (data/ is git-ignored; only the reported numbers enter the EXP-007 report)

Stage A (100k, no near-dup): prefilter -> license -> quality -> secrets
  -> exact dedup -> HumanEval+ decontamination -> hash split.
Stage B (50k, full pipeline incl. near-dup): measures the near-dup rate
  at scale and the peak RSS of the near-dup stage.
Checks (both stages): heldout-task isolation (prompt+tests, M4 check),
  split isolation, secret-kept = 0, manifest hygiene (no content key).
"""
import gzip
import hashlib
import json
import os
import resource
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_pipeline import (PipelineConfig, new_record, prefilter, license_ok,
                           quality_check, scan_secrets, exact_dedup, near_dedup,
                           assign_split, write_manifest, write_corpus)
from decontam import Decontaminator, load_fingerprints, normalize_code

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHARD = os.path.join(ROOT, "data", "audit", "shard1.json.gz")
OUTDIR = os.path.join(ROOT, "data", "audit_m2")
REV = "35a59fb025bc0a102f7d96eac09d145b896d487b"


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_rows(n):
    records = []
    with gzip.open(SHARD, "rt") as f:
        for i, line in enumerate(f):
            if i >= n:
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
    return records


def run_stages(records, cfg, dec, with_near_dup, tag):
    """Same stage code as run_pipeline, with per-stage RSS prints."""
    import time
    from collections import defaultdict
    t0 = time.time()
    stats = defaultdict(int)
    stats["discovered"] = len(records)
    print(f"[{tag}] discovered={len(records)} rss={rss_mb():.0f}MB", flush=True)
    stage1 = []
    for rec in records:
        reason = prefilter(rec, cfg)
        if reason:
            rec["excluded"] = True
            rec["exclusion_reason"] = reason
            stats["excluded_" + reason] += 1
        else:
            stage1.append(rec)
    stats["after_prefilter"] = len(stage1)
    print(f"[{tag}] after_prefilter={len(stage1)} rss={rss_mb():.0f}MB", flush=True)
    stage2 = []
    for rec in stage1:
        if not license_ok(rec, cfg):
            rec["excluded"] = True
            rec["exclusion_reason"] = f"license:{rec['license'] or 'unknown'}"
            stats["excluded_license"] += 1
        else:
            stage2.append(rec)
    stats["after_license"] = len(stage2)
    print(f"[{tag}] after_license={len(stage2)} rss={rss_mb():.0f}MB", flush=True)
    stage3 = []
    for rec in stage2:
        reason = quality_check(rec, cfg)
        if reason:
            rec["excluded"] = True
            rec["exclusion_reason"] = reason
            stats["excluded_" + reason] += 1
        else:
            stage3.append(rec)
    stats["after_quality"] = len(stage3)
    print(f"[{tag}] after_quality={len(stage3)} rss={rss_mb():.0f}MB", flush=True)
    stage4 = []
    for rec in stage3:
        if not cfg.scan_secrets:
            stage4.append(rec)
            continue
        hits = scan_secrets(rec)
        if hits and cfg.drop_on_secret:
            rec["excluded"] = True
            rec["secret_hits"] = hits
            rec["exclusion_reason"] = "possible_secret:" + hits[0]
            stats["excluded_secret"] += 1
            stats["secret_flagged"] += 1
        else:
            if hits:
                rec["secret_hits"] = hits
                stats["secret_flagged"] += 1
            stage4.append(rec)
    stats["after_secrets"] = len(stage4)
    print(f"[{tag}] after_secrets={len(stage4)} rss={rss_mb():.0f}MB", flush=True)
    stage5 = exact_dedup(stage4, stats)
    stats["after_exact_dedup"] = len(stage5)
    stats["exact_unique"] = len(stage5)
    print(f"[{tag}] after_exact_dedup={len(stage5)} rss={rss_mb():.0f}MB", flush=True)
    if with_near_dup:
        stage6 = near_dedup(stage5, cfg, stats)
        stats["after_near_dedup"] = len(stage6)
        print(f"[{tag}] after_near_dedup={len(stage6)} rss={rss_mb():.0f}MB", flush=True)
    else:
        stage6 = stage5
        stats["after_near_dedup"] = len(stage6)
        print(f"[{tag}] near-dup SKIPPED (see report) rss={rss_mb():.0f}MB", flush=True)
    stage7 = dec.apply(stage6, stats)
    stats["after_decontam"] = len([r for r in stage7 if not r["excluded"]])
    stage7 = [r for r in stage7 if not r["excluded"]]
    print(f"[{tag}] after_decontam={len(stage7)} rss={rss_mb():.0f}MB", flush=True)
    for rec in stage7:
        rec["split"] = assign_split(rec, cfg)
    stats["retained"] = len(stage7)
    stats["bytes_retained"] = sum(r["bytes"] for r in stage7)
    stats["config_fingerprint"] = cfg.fingerprint()
    stats["split_train"] = sum(1 for r in stage7 if r["split"] == "train")
    stats["split_val"] = sum(1 for r in stage7 if r["split"] == "val")
    stats["split_test"] = sum(1 for r in stage7 if r["split"] == "test")
    stats["elapsed_s"] = round(time.time() - t0, 2)
    return records, stage7, dict(stats)


def audit_checks(all_records, kept, tag):
    kept_recs = [r for r in all_records if not r["excluded"]]
    assert len(kept_recs) == len(kept)
    secret_kept = sum(1 for r in kept_recs if r.get("secret_hits"))
    sha_splits = {}
    for r in kept_recs:
        sha_splits.setdefault(r["sha256"], set()).add(r["split"])
    cross_split = sum(1 for v in sha_splits.values() if len(v) > 1)
    bench = json.load(open(os.path.join(ROOT, "benchmarks", "forge_heldout_v1.json")))
    corpus_hashes = {r["sha256"] for r in kept_recs}
    hits = []
    for t in bench["tasks"]:
        text = t["prompt"] + "\n" + "\n".join(t["tests"])
        if (hashlib.sha256(text.encode()).hexdigest() in corpus_hashes or
                ("N" + hashlib.sha256(normalize_code(text).encode()).hexdigest()) in corpus_hashes):
            hits.append(t["id"])
    print(f"[{tag}] secret_kept={secret_kept} cross_split={cross_split} heldout_hits={hits} "
          f"peak_rss={rss_mb():.0f}MB", flush=True)
    return {"secret_kept": secret_kept, "cross_split_shas": cross_split,
            "heldout_task_hits": hits, "peak_rss_mb": round(rss_mb(), 1)}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = PipelineConfig(max_files=100000)
    fp = load_fingerprints(os.path.join(ROOT, "benchmarks", "fingerprints.json"))
    # ---- Stage A: 100k, no near-dup (numbers that must survive) ----
    recs100 = load_rows(100000)
    print(f"[A] loaded {len(recs100)} rows rss={rss_mb():.0f}MB", flush=True)
    all100, kept100, stats100 = run_stages(recs100, cfg, Decontaminator(fp), False, "A")
    prov = json.load(open(os.path.join(ROOT, "data", "provenance.json")))
    write_manifest(all100, os.path.join(OUTDIR, "manifest.jsonl"), prov)
    write_corpus(kept100, os.path.join(OUTDIR, "corpus.jsonl"))
    with open(os.path.join(OUTDIR, "stats.json"), "w") as f:
        json.dump(stats100, f, indent=2, sort_keys=True)
    checks100 = audit_checks(all100, kept100, "A")
    manifest_keys = set()
    with open(os.path.join(OUTDIR, "manifest.jsonl")) as f:
        for line in f:
            manifest_keys.update(json.loads(line).keys())
    audit = {
        "source": "codeparrot/codeparrot-clean", "source_rev": REV,
        "shard": "file-000000000001.json.gz (100,000 rows)",
        "stage_A_100k_no_neardup": stats100, "stage_A_checks": checks100,
        "manifest_keys": sorted(manifest_keys),
        "manifest_has_content": "content" in manifest_keys,
    }
    with open(os.path.join(OUTDIR, "audit.json"), "w") as f:
        json.dump(audit, f, indent=2, sort_keys=True)
    print(f"[A] DONE retained={stats100['retained']} bytes={stats100['bytes_retained']} "
          f"elapsed={stats100['elapsed_s']}s", flush=True)
    # ---- Stage B: 45k, full pipeline incl. near-dup (rate + memory) ----
    # 45k rows = the pilot's corpus size: this also proves the pilot's
    # corpus-prep fits in RAM (10k survivors cost ~850MB in validation;
    # ~21k survivors project to ~4GB, inside Kaggle's ~13GB and this VM).
    cfg50 = PipelineConfig(max_files=45000)
    recs50 = load_rows(45000)
    print(f"[B] loaded {len(recs50)} rows rss={rss_mb():.0f}MB", flush=True)
    all50, kept50, stats50 = run_stages(recs50, cfg50, Decontaminator(fp), True, "B")
    checks50 = audit_checks(all50, kept50, "B")
    with open(os.path.join(OUTDIR, "neardup_50k.json"), "w") as f:
        json.dump({"stats": stats50, "checks": checks50}, f, indent=2, sort_keys=True)
    print(f"[B] DONE retained={stats50['retained']} near_dups={stats50.get('near_dups')} "
          f"peak_rss={rss_mb():.0f}MB", flush=True)


if __name__ == "__main__":
    main()
