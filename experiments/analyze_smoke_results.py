#!/usr/bin/env python3
"""Review tool for the Kaggle GPU smoke-run results (EXP-005, Gate E validation).

Reads the gpu_smoke_results.json that the FORGE M5 smoke notebook writes to
/kaggle/working/ and INDEPENDENTLY verifies it -- schema, commit pin,
hardware sanity, tok/s self-consistency, loss sanity vs the M3 CPU reference,
and a from-scratch MFU + 125M-projection recomputation (not trusting the
notebook's own arithmetic).

Usage:
    python experiments/analyze_smoke_results.py /path/to/gpu_smoke_results.json

Exit 0 = all checks pass (warnings allowed). Exit 1 = a check failed.

This script is a review aid only: it never modifies the input file and it
never touches experiment evidence. torch-free (uses src/model.py arithmetic).
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from model import ForgeConfig  # noqa: E402  (torch-free: pure arithmetic)

# The GitHub snapshot commit the current smoke notebook pins. Bump this when
# the notebook is re-pinned (a mismatch alone only means a newer/older
# notebook ran -- the binding check is pinned_commit == actual_head).
KNOWN_PIN = "fa631226c43f2ce24df5c6793ec8491ec5baf4f3"

# FP32 peaks (training is FP32, no AMP) -- must agree with the notebook.
FP32_PEAK = {"T4": 8.1e12, "P100": 9.3e12}

# Expected run configs, mirroring the notebook's run_cfg (steps/batch/seq/gpus).
EXPECTED = {
    "A-tiny-1M": {"steps": 40, "batch": 8, "seq": 64, "gpus": 1,
                  "d_model": 128, "n_layers": 4},
    "B-25M": {"steps": 300, "batch": 16, "seq": 512, "gpus": 1,
              "d_model": 512, "n_layers": 8},
    "C-25M-ddp2": {"steps": 40, "batch": 16, "seq": 512, "gpus": 2,
                   "d_model": 512, "n_layers": 8},
}

# M3 CPU reference for Run A (40-step tiny smoke, bit-identical on CPU).
# GPU runs are NOT expected to be bit-identical (CUDA nondeterminism), so the
# band below is intentionally loose -- it catches divergence/NaN, not noise.
M3_REF = {"final_train_loss": 17.793315887451172,
          "final_val_loss": 18.236953735351562}
# Loss sanity is NOT an absolute band: on this fixture a bigger model trained
# longer memorizes the templates (25M/300 steps -> ~0.17 is success, not
# failure). Check finiteness/positivity + train/val agreement instead.

# EXP-003 Gate E assumption under test (125M v0, 2.5B tokens).
GATE_E_BAND_T4H = (23.0, 32.0)
GATE_E_MFU_BAND = (0.25, 0.35)


def flops_per_token(d_model: int, n_layers: int, seq: int = 512):
    """FLOPs/token ~ 6N + 12*L*d*S (Kaplan 6N + attention term), +/-10% band.

    Mirrors the notebook's methodology so the recomputation is comparable;
    N comes from the exact config arithmetic in src/model.py.
    """
    cfg = ForgeConfig(vocab_size=256, d_model=d_model, n_layers=n_layers,
                      n_heads=8, n_kv_heads=4, max_seq_len=seq + 1)
    n = cfg.param_count()["total"]
    return 6 * n + 12 * n_layers * d_model * seq, n


class Report:
    def __init__(self):
        self.oks, self.warns, self.fails = [], [], []

    def ok(self, msg): self.oks.append(msg)
    def warn(self, msg): self.warns.append(msg)
    def fail(self, msg): self.fails.append(msg)

    def passed(self): return not self.fails

    def render(self):
        for m in self.oks: print(f"  [ok]   {m}")
        for m in self.warns: print(f"  [warn] {m}")
        for m in self.fails: print(f"  [FAIL] {m}")
        print(f"\n{len(self.oks)} ok, {len(self.warns)} warnings, "
              f"{len(self.fails)} failures")


def main(path: str) -> int:
    rep = Report()
    try:
        with open(path) as f:
            p = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"cannot read results JSON: {e}")
        return 2

    # --- identity ---------------------------------------------------------
    if p.get("notebook") == "forge_m5_gpu_smoke":
        rep.ok(f"notebook tag = {p['notebook']!r}")
    else:
        rep.fail(f"unexpected notebook tag: {p.get('notebook')!r}")

    pinned, actual = p.get("pinned_commit"), p.get("actual_head")
    if pinned == KNOWN_PIN:
        rep.ok(f"pinned_commit matches the fixed-notebook pin ({KNOWN_PIN[:8]}...)")
    else:
        rep.warn(f"pinned_commit {str(pinned)[:12]}... != fixed-notebook pin "
                 f"{KNOWN_PIN[:12]}... (older/different notebook?)")
    if actual and pinned and str(actual).startswith(str(pinned)[:8]):
        rep.ok(f"actual_head {str(actual)[:12]}... matches pinned commit")
    else:
        rep.fail(f"actual_head {str(actual)[:12]}... does not match pinned "
                 f"{str(pinned)[:12]}... -- provenance unverified")

    # --- hardware ----------------------------------------------------------
    hw = p.get("hardware", {}) or {}
    names = hw.get("gpu_names", []) or []
    n_gpu = hw.get("gpu_count", 0)
    if n_gpu >= 1 and names:
        rep.ok(f"hardware: {n_gpu}x {', '.join(names)}")
    else:
        rep.fail(f"no GPUs reported (count={n_gpu}, names={names})")
    peak = hw.get("fp32_peak_flops_used")
    matched_peak = None
    for nm in names:
        for k, v in FP32_PEAK.items():
            if k in nm:
                matched_peak = v
    if matched_peak and peak:
        if math.isclose(peak, matched_peak, rel_tol=0.05):
            rep.ok(f"FP32 peak {peak/1e12:.1f} TFLOPS matches {names}")
        else:
            rep.fail(f"FP32 peak {peak} inconsistent with GPU names {names}")
    elif peak:
        rep.warn(f"unrecognized GPU names {names}; peak {peak} taken at face value")
    else:
        rep.fail("missing fp32_peak_flops_used -- MFU cannot be recomputed")

    # --- runs ---------------------------------------------------------------
    runs = p.get("runs", {}) or {}
    mfu_b = None  # recomputed Run B MFU, for the Gate E projection
    for tag, exp in EXPECTED.items():
        r = runs.get(tag)
        if not r or r.get("status") != "ok":
            if tag == "C-25M-ddp2":
                rep.warn("Run C (2-GPU DDP) not run -- single-GPU results still stand")
            else:
                rep.fail(f"run {tag}: status={r.get('status') if r else 'missing'}")
            continue
        cfg = r.get("config", {}) or {}
        for k in ("steps", "batch", "seq", "gpus"):
            if cfg.get(k) != exp[k]:
                rep.warn(f"{tag}: config {k}={cfg.get(k)} != notebook default {exp[k]}")
        tps = r.get("tok_per_s")
        el = r.get("elapsed_s")
        if tps and tps > 0 and el and el > 0:
            # self-consistency: tok/s ~= steps*batch*seq*gpus / elapsed
            # (train.py: tok_per_s = tokens_seen*world_size / elapsed)
            expect = exp["steps"] * exp["batch"] * exp["seq"] * exp["gpus"] / el
            if math.isclose(tps, expect, rel_tol=0.20):
                rep.ok(f"{tag}: tok/s={tps:.0f} self-consistent "
                       f"(config-implied {expect:.0f})")
            else:
                rep.fail(f"{tag}: tok/s={tps:.0f} vs config-implied {expect:.0f} "
                         f"-- arithmetic mismatch")
        else:
            rep.fail(f"{tag}: bad tok/s ({tps}) or elapsed ({el})")
        for lk in ("final_train_loss", "final_val_loss"):
            lv = r.get(lk)
            if isinstance(lv, (int, float)) and math.isfinite(lv) and lv > 0:
                rep.ok(f"{tag}: {lk}={lv:.4g} finite and positive")
            else:
                rep.fail(f"{tag}: {lk}={lv} not finite/positive "
                         f"(divergence or failed run?)")
        # train/val agreement: a large gap signals a real problem; a low
        # absolute loss is memorization of this fixture, not failure.
        tl, vl = r.get("final_train_loss"), r.get("final_val_loss")
        if isinstance(tl, (int, float)) and isinstance(vl, (int, float)) \
                and tl > 0 and vl > 0:
            ratio = max(tl, vl) / min(tl, vl)
            if ratio < 2.0:
                rep.ok(f"{tag}: train/val agree (ratio {ratio:.2f})")
            else:
                rep.fail(f"{tag}: train/val disagree (ratio {ratio:.2f})")
        mem = r.get("peak_gpu_mem_mb")
        if isinstance(mem, (int, float)) and mem and mem > 0:
            rep.ok(f"{tag}: peak_gpu_mem_mb={mem:.0f}")

        # independent MFU recomputation (do not trust the notebook's number).
        # tok/s is aggregate across ranks, so the denominator is peak x gpus:
        # dividing 2-GPU throughput by a single GPU's peak overstates MFU 2x.
        if peak and tps:
            fpt, n_exact = flops_per_token(exp["d_model"], exp["n_layers"])
            peak_total = peak * exp["gpus"]
            mfu = tps * fpt / peak_total
            nb_mfu = r.get("mfu_fp32")
            line = (f"{tag}: independent MFU(fp32)={mfu:.3f} "
                    f"(N={n_exact/1e6:.2f}M, {tps:.0f} tok/s x {fpt/1e9:.2f} GFLOP/tok "
                    f"/ {peak_total/1e12:.1f} TFLOPS over {exp['gpus']} GPU(s))")
            if nb_mfu is not None and math.isclose(mfu, nb_mfu, rel_tol=0.05):
                rep.ok(line + f" -- agrees with notebook ({nb_mfu})")
            else:
                rep.warn(line + f" -- notebook said {nb_mfu}")
            if 0.0 < mfu < 1.0:
                rep.ok(f"{tag}: MFU in (0,1) -- physically plausible")
            else:
                rep.fail(f"{tag}: MFU={mfu:.3f} implausible")
            if tag == "B-25M":
                mfu_b = mfu

    # Run A vs the M3 CPU reference (loose: same ballpark, not bit-identical)
    ra = runs.get("A-tiny-1M", {}) or {}
    if ra.get("status") == "ok":
        for lk, ref in M3_REF.items():
            lv = ra.get(lk)
            if isinstance(lv, (int, float)) and abs(lv - ref) / ref < 0.5:
                rep.ok(f"Run A {lk}={lv:.2f} within 50% of M3 CPU ref {ref:.2f}")
            else:
                rep.warn(f"Run A {lk}={lv} far from M3 CPU ref {ref:.2f} "
                         f"(GPU nondeterminism or a real problem -- inspect)")

    # --- Gate E projection ----------------------------------------------------
    print("\n--- Gate E projection (125M / 2.5B tokens) ---")
    if mfu_b and 0.0 < mfu_b < 1.0 and peak:
        fpt125, n125 = flops_per_token(768, 18)  # FORGE-125M config per notebook
        hrs = fpt125 * 2.5e9 / (peak * mfu_b) / 3600
        lo, hi = GATE_E_BAND_T4H
        verdict = "INSIDE" if lo <= hrs <= hi else "OUTSIDE"
        print(f"  FORGE-125M exact params : {n125/1e6:.1f}M")
        print(f"  measured MFU (Run B)    : {mfu_b:.3f} "
              f"(EXP-003 assumed {GATE_E_MFU_BAND[0]:.0%}-{GATE_E_MFU_BAND[1]:.0%})")
        print(f"  projected GPU-hours     : ~{hrs:.1f}  [{verdict} EXP-003 band "
              f"{lo:.0f}-{hi:.0f} T4-hours]")
        print("  (+/-10% FLOPs-methodology band; short-run measurement noise not included)")
    else:
        print("  cannot project: Run B MFU unavailable")

    print("\n--- checks ---")
    rep.render()
    return 0 if rep.passed() else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
