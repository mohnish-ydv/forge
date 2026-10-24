#!/usr/bin/env python3
"""Review tool for the Kaggle AMP benchmark results (EXP-006).

Reads the amp_bench_results.json that notebooks/forge_m6_amp_bench.ipynb
writes to /kaggle/working/ and INDEPENDENTLY verifies it -- schema, commit
pin, hardware sanity, tok/s self-consistency, the comparison numbers
(speedup, both MFUs), the 125M projection recomputation, loss/scaler
stability, and the resume test -- then issues a PASS/FAIL verdict against
the EXP-006 section 8 criteria (not trusting the notebook's arithmetic).

Usage:
    python experiments/analyze_amp_results.py /path/to/amp_bench_results.json

Exit 0 = all checks pass and the AMP decision criteria are met (warnings
allowed). Exit 1 = a check failed or the verdict is FAIL.

This script is a review aid only: it never modifies the input file and it
never touches experiment evidence. torch-free (uses src/model.py arithmetic).

Verdict criteria (this tool's explicit reading of EXP-006 section 8):
  PASS requires, on the same T4: steady FP16 tok/s materially above FP32
  (speedup >= 1.3x), losses finite, fp16 skipped_steps <= 5% of steps, the
  fresh-process resume test passing with scaler restored, and the
  125M/2.5B planning projection inside the ~30 GPU-h/week-feasible band.
  FAIL if fp16 is barely faster (speedup < 1.1x), most steps skipped
  (>50%), repeated non-finite losses/grads, or the resume test fails.
  Anything in between is CONDITIONAL (mixed signals, needs human review).
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from model import ForgeConfig  # noqa: E402  (torch-free: pure arithmetic)

# The implementation commit the benchmark notebook pins. Bump when the
# notebook is re-pinned (a mismatch alone only means a newer/older notebook
# ran -- the binding check is pinned_commit == actual_head).
KNOWN_PIN = "7d3e58ff4eb48853e2107d74ab4a8ad2d8874f09"

FP32_PEAK, FP16_PEAK = 8.1e12, 65e12  # T4; must agree with the notebook

# Expected single-GPU run config, mirroring the notebook's benchmark cell.
EXP_STEPS, EXP_WARMUP, EXP_BATCH, EXP_SEQ = 300, 20, 16, 512
EXP_CFG = {"d_model": 512, "n_layers": 8, "n_heads": 8, "n_kv_heads": 4}
N25 = 25_305_600  # exact param count asserted by the notebook itself

# Loss sanity: on this toy fixture a 25M/300-step run memorizes the
# templates (loss ~0.17 is success). Check finiteness/positivity and that
# fp16 tracks fp32, not an absolute band.
FP16_VS_FP32_LOSS_RATIO_MAX = 3.0

# Verdict thresholds (see docstring).
SPEEDUP_PASS, SPEEDUP_FAIL = 1.3, 1.1
SKIPPED_PASS_FRAC, SKIPPED_FAIL_FRAC = 0.05, 0.50
PLANNING_HOURS_FEASIBLE = 32.0  # ~one Kaggle weekly quota (EXP-003 band)


def flops_per_token(n, n_layers, d_model, seq):
    """FLOPs/token ~ 6N + 12*L*d*S. Mirrors the notebook's methodology."""
    return 6 * n + 12 * n_layers * d_model * seq


def verify_params():
    cfg = ForgeConfig(vocab_size=256, d_model=EXP_CFG["d_model"],
                      n_layers=EXP_CFG["n_layers"], n_heads=EXP_CFG["n_heads"],
                      n_kv_heads=EXP_CFG["n_kv_heads"], max_seq_len=EXP_SEQ + 1)
    return cfg.param_count()["total"]


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

    if p.get("status") != "ok":
        rep.fail(f"status={p.get('status')!r} -- benchmark not completed")
        rep.render()
        return 1

    # --- identity ---------------------------------------------------------
    if p.get("notebook") == "forge_m6_amp_bench":
        rep.ok(f"notebook tag = {p['notebook']!r}")
    else:
        rep.fail(f"unexpected notebook tag: {p.get('notebook')!r}")

    pinned, actual = p.get("pinned_commit"), p.get("actual_head")
    if pinned == KNOWN_PIN:
        rep.ok(f"pinned_commit matches the AMP-notebook pin ({KNOWN_PIN[:8]}...)")
    else:
        rep.warn(f"pinned_commit {str(pinned)[:12]}... != AMP-notebook pin "
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
    vram = hw.get("vram_total_mb", []) or []
    if n_gpu >= 1 and names and any("T4" in n for n in names):
        rep.ok(f"hardware: {n_gpu}x {', '.join(names)}")
    else:
        rep.fail(f"unexpected hardware (count={n_gpu}, names={names}) -- "
                 f"benchmark targets one T4")
    for peak, label, want in (("fp32_peak_flops_used", "FP32", FP32_PEAK),
                              ("fp16_peak_flops_used", "FP16", FP16_PEAK)):
        got = hw.get(peak)
        if got and math.isclose(got, want, rel_tol=0.01):
            rep.ok(f"{label} peak {got/1e12:.1f} TFLOPS matches T4 {label}")
        else:
            rep.fail(f"{label} peak {got} != expected {want/1e12:.1f} TFLOPS")

    # --- runs ---------------------------------------------------------------
    runs = p.get("runs", {}) or {}
    s32, s16 = runs.get("fp32"), runs.get("fp16")
    if not s32 or not s16:
        rep.fail("runs.fp32 and/or runs.fp16 missing -- both runs required")
        rep.render()
        return 1
    for tag, s, prec in (("fp32", s32, "fp32"), ("fp16", s16, "fp16")):
        if s.get("precision") == prec:
            rep.ok(f"run {tag}: precision label = {prec!r}")
        else:
            rep.fail(f"run {tag}: precision={s.get('precision')!r} "
                     f"(evidence mislabeled?)")
        for k, v in (("steps", EXP_STEPS), ("warmup_steps", EXP_WARMUP),
                     ("ddp", False), ("world_size", 1)):
            if s.get(k) != v:
                rep.fail(f"run {tag}: {k}={s.get(k)} != expected {v}")
        # evidence separation: FP32 summary must carry the neutral values
        if tag == "fp32":
            if s.get("scaler_final_scale") is None and s.get("skipped_steps") == 0:
                rep.ok("run fp32: scaler null, skipped_steps 0 (not AMP evidence)")
            else:
                rep.warn("run fp32: scaler/skipped fields non-neutral")
        # token accounting: steps*batch*seq (world_size folds into tokens_seen
        # for DDP; these are single-GPU runs so world_size=1)
        expect_tokens = EXP_STEPS * EXP_BATCH * EXP_SEQ
        if s.get("tokens_seen") == expect_tokens:
            rep.ok(f"run {tag}: tokens_seen={expect_tokens:,} matches config")
        else:
            rep.fail(f"run {tag}: tokens_seen={s.get('tokens_seen')} != "
                     f"{expect_tokens} implied by config")
        # overall tok/s self-consistency: tokens_seen / elapsed
        el, tps = s.get("elapsed_s"), s.get("tok_per_s")
        if el and tps and tps > 0:
            expect = expect_tokens / el
            if math.isclose(tps, expect, rel_tol=0.05):
                rep.ok(f"run {tag}: tok_per_s={tps:.0f} self-consistent")
            else:
                rep.fail(f"run {tag}: tok_per_s={tps} vs implied {expect:.0f}")
        else:
            rep.fail(f"run {tag}: bad elapsed ({el}) or tok/s ({tps})")
        # steady region must exist and beat the overall average (warmup is
        # the slow part and validation time is in elapsed, not in steady)
        st = s.get("steady_tok_per_s")
        if st and st > 0:
            rep.ok(f"run {tag}: steady_tok_per_s={st:.0f} (post-warmup)")
            if tps and st < tps:
                rep.warn(f"run {tag}: steady {st:.0f} < overall {tps:.0f} "
                         f"-- warmup exclusion should speed it up")
        else:
            rep.fail(f"run {tag}: missing steady_tok_per_s (warmup >= steps?)")
        for lk in ("final_train_loss", "final_val_loss"):
            lv = s.get(lk)
            if isinstance(lv, (int, float)) and math.isfinite(lv) and lv > 0:
                rep.ok(f"run {tag}: {lk}={lv:.4g} finite and positive")
            else:
                rep.fail(f"run {tag}: {lk}={lv} not finite/positive")
        mem = s.get("peak_gpu_mem_mb") or 0
        mem_r = s.get("peak_gpu_mem_reserved_mb") or 0
        if mem > 0 and (not vram or mem < vram[0]):
            rep.ok(f"run {tag}: peak alloc {mem:.0f} MB, reserved {mem_r:.0f} MB")
        else:
            rep.fail(f"run {tag}: peak memory {mem} MB exceeds VRAM {vram}")
    # fp16 must not balloon memory vs fp32
    if s16.get("peak_gpu_mem_mb") and s32.get("peak_gpu_mem_mb"):
        if s16["peak_gpu_mem_mb"] <= s32["peak_gpu_mem_mb"] * 1.1:
            rep.ok(f"fp16 peak mem {s16['peak_gpu_mem_mb']:.0f} MB <= "
                   f"fp32 {s32['peak_gpu_mem_mb']:.0f} MB (no blowup)")
        else:
            rep.warn("fp16 uses MORE memory than fp32")

    # --- param count (exact, from src/model.py) -------------------------------
    n = verify_params()
    if n == N25:
        rep.ok(f"param count {n:,} matches the notebook's asserted 25,305,600")
    else:
        rep.fail(f"param count {n:,} != notebook's 25,305,600 -- MFU basis off")

    # --- comparison recomputation --------------------------------------------
    cmp_ = p.get("comparison", {}) or {}
    st32, st16 = s32["steady_tok_per_s"], s16["steady_tok_per_s"]
    fpt25 = flops_per_token(n, EXP_CFG["n_layers"], EXP_CFG["d_model"], EXP_SEQ)
    speedup = st16 / st32
    if math.isclose(cmp_.get("steady_speedup_fp16_vs_fp32", 0), round(speedup, 3),
                    rel_tol=1e-3):
        rep.ok(f"speedup {speedup:.3f}x matches notebook's number")
    else:
        rep.fail(f"speedup recomputed {speedup:.3f}x != notebook's "
                 f"{cmp_.get('steady_speedup_fp16_vs_fp32')}")
    mfu32 = st32 * fpt25 / FP32_PEAK
    mfu16 = st16 * fpt25 / FP16_PEAK
    if cmp_.get("scope", "").startswith("per-GPU"):
        rep.ok("comparison scope = per-GPU (single-GPU runs)")
    else:
        rep.fail(f"comparison scope={cmp_.get('scope')!r} -- must be per-GPU")
    for label, mine, theirs in (("mfu_fp32_vs_fp32_peak", mfu32, cmp_.get("mfu_fp32_vs_fp32_peak")),
                                ("mfu_fp16_vs_fp16_peak", mfu16, cmp_.get("mfu_fp16_vs_fp16_peak"))):
        if theirs is not None and math.isclose(theirs, round(mine, 4), rel_tol=1e-3):
            rep.ok(f"independent {label}={mine:.4f} matches")
        else:
            rep.fail(f"{label} recomputed {mine:.4f} != notebook's {theirs}")

    # --- fp16 stability vs fp32 -------------------------------------------------
    tl32, tl16 = s32["final_train_loss"], s16["final_train_loss"]
    if tl16 / tl32 <= FP16_VS_FP32_LOSS_RATIO_MAX:
        rep.ok(f"fp16 train loss {tl16:.4g} tracks fp32 {tl32:.4g}")
    else:
        rep.fail(f"fp16 train loss {tl16:.4g} diverged from fp32 {tl32:.4g}")
    skipped = s16.get("skipped_steps", 0) or 0
    nloss, ngrad = s16.get("nonfinite_losses", 0) or 0, s16.get("nonfinite_grads", 0) or 0
    if skipped / EXP_STEPS <= SKIPPED_PASS_FRAC:
        rep.ok(f"fp16 skipped {skipped}/{EXP_STEPS} "
               f"({100*skipped/EXP_STEPS:.1f}% -- small minority)")
    elif skipped / EXP_STEPS > SKIPPED_FAIL_FRAC:
        rep.fail(f"fp16 skipped {skipped}/{EXP_STEPS} -- most steps skipped")
    else:
        rep.warn(f"fp16 skipped {skipped}/{EXP_STEPS} -- elevated")
    if nloss == 0 and ngrad == 0:
        rep.ok("fp16: zero non-finite losses/grads")
    elif (nloss + ngrad) / EXP_STEPS > 0.10:
        rep.fail(f"fp16: repeated non-finite (losses={nloss}, grads={ngrad})")
    else:
        rep.warn(f"fp16: some non-finite (losses={nloss}, grads={ngrad})")
    scale = s16.get("scaler_final_scale")
    if isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0:
        rep.ok(f"fp16 scaler final scale={scale:.4g} (finite, active)")
    else:
        rep.fail(f"fp16 scaler final scale={scale} -- scaler unhealthy")

    # --- resume test ------------------------------------------------------------
    rs = p.get("resume_test", {}) or {}
    if rs.get("status") == "ok":
        rep.ok("resume_test status=ok")
    else:
        rep.fail(f"resume_test status={rs.get('status')!r}")
    if rs.get("precision") == "fp16" and rs.get("steps") == 40:
        rep.ok("resume_test: 40 steps, fp16 precision")
    else:
        rep.fail(f"resume_test params: precision={rs.get('precision')!r}, "
                 f"steps={rs.get('steps')}")
    if rs.get("scaler_restored") is True:
        rep.ok("resume_test: scaler restored from checkpoint")
    else:
        rep.fail(f"resume_test scaler_restored={rs.get('scaler_restored')}")
    if rs.get("mismatch_rejected") is True:
        rep.ok("resume_test: cross-precision resume rejected")
    else:
        rep.fail(f"resume_test mismatch_rejected={rs.get('mismatch_rejected')}")

    # --- 125M/2.5B projection recomputation --------------------------------------
    proj = p.get("projection_125m_2p5B", {}) or {}
    fpt125 = flops_per_token(125_000_000, 18, 768, 512)  # config-level approx
    n125 = ForgeConfig(vocab_size=256, d_model=768, n_layers=18, n_heads=12,
                       n_kv_heads=3, max_seq_len=513).param_count()["total"]
    fpt125 = flops_per_token(n125, 18, 768, 512)
    base = 2.5e9 * fpt125 / (st16 * fpt25) / 3600
    for label, mine, theirs, mult in (("hours_base_from_measured_amp", base,
                                       proj.get("hours_base_from_measured_amp"), 1.0),
                                      ("hours_with_overheads_15pct", base * 1.15,
                                       proj.get("hours_with_overheads_15pct"), 1.15),
                                      ("hours_planning_50pct", base * 1.5,
                                       proj.get("hours_planning_50pct"), 1.5)):
        if theirs is not None and math.isclose(theirs, round(mine, 1), abs_tol=0.1):
            rep.ok(f"projection {label}={theirs}h matches recomputation")
        else:
            rep.fail(f"projection {label} recomputed {mine:.1f}h != {theirs}h")
    rep.ok(f"125M param count from config: {n125:,} (basis documented)")

    # --- verdict -------------------------------------------------------------------
    print()
    rep.render()
    print()
    resume_ok = (rs.get("status") == "ok" and rs.get("scaler_restored") is True
                 and rs.get("mismatch_rejected") is True)
    losses_ok = all(isinstance(s[k], (int, float)) and math.isfinite(s[k]) and s[k] > 0
                    for s in (s32, s16) for k in ("final_train_loss", "final_val_loss"))
    skipped_ok = skipped / EXP_STEPS <= SKIPPED_PASS_FRAC
    skipped_bad = skipped / EXP_STEPS > SKIPPED_FAIL_FRAC
    unstable = skipped_bad or (nloss + ngrad) / EXP_STEPS > 0.10
    planning = proj.get("hours_planning_50pct")

    fail_reasons = []
    if speedup < SPEEDUP_FAIL:
        fail_reasons.append(f"speedup {speedup:.2f}x < 1.1x (barely faster)")
    if unstable:
        fail_reasons.append("fp16 unstable (repeated skips/non-finite)")
    if not resume_ok:
        fail_reasons.append("resume test did not pass")
    if fail_reasons:
        print("VERDICT: FAIL -- " + "; ".join(fail_reasons))
        print("M5 plan must change (smaller model, fewer tokens, different "
              "compute strategy) -- do not quietly proceed at FP32.")
        return 1
    if not rep.passed():
        print("VERDICT: CONDITIONAL -- checks failed; needs human review.")
        return 1
    feas_ok = planning is not None and planning <= PLANNING_HOURS_FEASIBLE
    if speedup >= SPEEDUP_PASS and losses_ok and skipped_ok and resume_ok and feas_ok:
        print(f"VERDICT: PASS -- AMP viable: {speedup:.2f}x measured speedup, "
              f"stable, resume ok, 125M planning figure {planning}h inside "
              f"~{PLANNING_HOURS_FEASIBLE:.0f} GPU-h/week feasibility.")
        return 0
    reasons = []
    if speedup < SPEEDUP_PASS:
        reasons.append(f"speedup {speedup:.2f}x below 1.3x")
    if not feas_ok:
        reasons.append(f"planning {planning}h outside feasibility")
    print("VERDICT: CONDITIONAL -- " + ("; ".join(reasons) if reasons else
          "mixed signals") + ". Needs human review before M5.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python experiments/analyze_amp_results.py "
              "/path/to/amp_bench_results.json")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
