"""Run FORGE-Heldout v1 with a completions provider; print + save results.

Usage:
  python benchmarks/run_heldout.py --baseline empty [--split dev|heldout|all]
                                   [--out results/heldout_empty.json]

Baselines (deterministic, fixed strings):
  empty : body is `pass`            (B0 floor — must score 0.0)
  none  : body is `return None`     (naive stub — must score 0.0)
  oracle: reference solutions from build_heldout.py (benchmark-correctness check,
          NOT a model result — must score 1.0)

Reproducibility: baselines are fixed strings; the harness executes each
candidate in a fresh subprocess with a 10s wall-clock timeout. No sampling,
no seeds involved. Re-running the same command reproduces the JSON exactly
(modulo timing fields).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_harness import Problem, evaluate  # noqa: E402
import build_heldout  # noqa: E402  (quarantined oracles live here, never in training data)

BENCH_PATH = Path(__file__).parent / "forge_heldout_v1.json"


def load_benchmark() -> dict:
    bench = json.loads(BENCH_PATH.read_text())
    # integrity: recompute benchmark hash
    digest = bench.pop("benchmark_sha256")
    canon = json.dumps(bench, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canon.encode()).hexdigest() == digest, "benchmark file tampered"
    bench["benchmark_sha256"] = digest
    return bench


def completions_for(baseline: str, tasks: list[dict]) -> dict[str, str]:
    if baseline == "empty":
        return {t["id"]: "    pass" for t in tasks}
    if baseline == "none":
        return {t["id"]: "    return None" for t in tasks}
    if baseline == "oracle":
        return {t["id"]: build_heldout.ORACLES[t["id"]] for t in tasks}
    raise ValueError(f"unknown baseline {baseline}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["empty", "none", "oracle"], required=True)
    ap.add_argument("--split", choices=["dev", "heldout", "all"], default="all")
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bench = load_benchmark()
    tasks = [t for t in bench["tasks"] if args.split == "all" or t["split"] == args.split]
    problems = [Problem(name=t["id"], prompt=t["prompt"], tests=t["tests"],
                        entry_point=t["entry_point"]) for t in tasks]
    completions = completions_for(args.baseline, tasks)

    t0 = time.time()
    res = evaluate(problems, completions, timeout=args.timeout)
    elapsed = time.time() - t0

    print(f"benchmark={bench['version']} sha={bench['benchmark_sha256'][:16]}… "
          f"baseline={args.baseline} split={args.split}")
    print(f"pass@1 = {res['passed']}/{res['total']} = {res['pass_at_1']:.3f} "
          f"({elapsed:.1f}s wall)")
    print(f"error funnel: {res['error_funnel']}")
    for r in res["results"]:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.problem}: {r.error_class}"
              + (f" — {r.detail[:70]}" if not r.passed else ""))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": bench["version"],
            "benchmark_sha256": bench["benchmark_sha256"],
            "baseline": args.baseline,
            "split": args.split,
            "timeout": args.timeout,
            "elapsed_s": round(elapsed, 2),
            "pass_at_1": res["pass_at_1"],
            "passed": res["passed"],
            "total": res["total"],
            "error_funnel": res["error_funnel"],
            "per_task": [{"id": r.problem, "passed": r.passed,
                          "error_class": r.error_class, "detail": r.detail}
                         for r in res["results"]],
        }
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
