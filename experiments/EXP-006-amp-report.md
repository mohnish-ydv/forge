# EXP-006 — AMP experiment (FP16) and verified throughput

Date: 2026-10-24. Repo: ~/workspace/forge.
Follows EXP-005 (Kaggle T4 ×2 smoke run): Run B measured MFU 30.9% (FP32),
but the honest FP32 projection for 125M/2.5B tokens is ~218 GPU-hours —
not viable on Kaggle's ~30 GPU-h/week quota. This experiment tests whether
FP16 mixed precision fixes that, without breaking stability, checkpoints,
or the FP32 baseline.

## Status: RUN — real Kaggle GPU results below (section 7)

Run 2026-10-24 on 2× Tesla T4 (single-GPU benchmark runs), notebook pinned
to GitHub snapshot `7d3e58ff` (the AMP implementation). Raw results
archived at `experiments/amp_bench_results.json` (as uploaded by Mohnish)
and independently recomputed (pin, config equality, speedup, both MFUs,
projection, budget arithmetic — all agree). **Decision: CONDITIONAL PASS**
(section 8). FP32 losses reproduce EXP-005 Run B bit-for-bit, which
cross-validates both runs.

## 1. What changed

| file | change |
|---|---|
| `src/train.py` | `--precision fp32\|fp16` (default fp32); `--warmup-steps`; CUDA autocast + GradScaler; skipped-step counting via optimizer-step wrapper; scaler state in checkpoints; resume precision-compatibility rejection; CPU-FP16 rejection; steady-state (post-warmup, CUDA-synchronized) tok/s; peak reserved memory; skipped/non-finite/scaler fields in summary |
| `src/model.py` | RMSNorm computes in fp32 and casts back (no-op for fp32 inputs → FP32 baseline bit-identical) |
| `tests/test_amp.py` | 14 tests: precision validation, scaler round-trip, checkpoint scaler plumbing, resume-precision decisions, exact FP32 CPU smoke, evidence separation, DDP rank-0 writes |
| `notebooks/forge_m6_amp_bench.ipynb` | Fair FP32-vs-FP16 Kaggle benchmark (single GPU), AMP resume test, results JSON writer |
| `experiments/amp_bench_results.template.json` | Machine-readable schema for the benchmark output (status NOT_RUN until measured) |

## 2. Numerical policy — what runs in which precision

Blindly casting everything to FP16 is how AMP runs diverge. The policy here:

- **Matmuls / linears:** FP16 under autocast (tensor cores). This is where
  the speedup comes from.
- **RMSNorm:** FP32 explicitly (changed in `model.py`). Sum-of-squares in
  FP16 overflows above 256² accumulations; for FP32 inputs the change is
  a no-op, so the FP32 baseline is bit-identical (proven by the exact-smoke
  test, section 3).
- **Attention (SDPA):** FP16 matmuls with FP32 softmax accumulation
  (kernel behavior). No code change; documented, not assumed.
- **Logits:** FP16 linear output, cast to FP32 for the loss — cross-entropy
  is on autocast's FP32 list. Loss is therefore computed in FP32.
- **RoPE:** frequencies computed in FP32; rotation products are bounded
  [-1,1] factors, so FP16 products carry no overflow risk.
- **Optimizer / master weights:** FP32 always (autocast never casts
  parameters). GradScaler scales the FP32-computed loss, unscales before
  clipping, and skips the optimizer step when inf/nan is found.
- **Scheduler on skipped steps:** advances unconditionally (PyTorch docs
  pattern). Skipped steps are reported in the summary, so the effect is
  visible, not hidden.

## 3. Tests (this VM, CPU-only unless noted)

Exact commands:

```
.venv/bin/python tests/test_amp.py        # 14 AMP tests
.venv/bin/python tests/test_train.py      # 10 training tests (1 CUDA-only skipped)
.venv/bin/python tests/test_pipeline.py   # 26
.venv/bin/python tests/test_decontam.py   # 6
.venv/bin/python tests/test_heldout.py    # 21
```

Results: **77 tests, all passing** (1 skipped: the CUDA-only param_hash
test, which needs a GPU). Highlights:

- Precision validation: `--precision bf16` rejected by argparse;
  `--precision fp16` on CPU rejected with "requires a CUDA device"
  (both `--device cpu` and `--device auto` on this CPU-only machine).
- Scaler round-trip: `GradScaler.state_dict()` → fresh scaler → equal
  dicts (works on CPU with an expected "CUDA not available, disabling"
  warning; serialization, not execution, is what is tested).
- Checkpoint plumbing: FP32 (v1-style, scaler=None) checkpoint loads with
  a scaler passed and leaves it untouched; scaler state survives a
  save/load cycle intact.
- Resume decisions (pure function, all cases): fp32↔fp32 ok, fp16↔fp16
  with scaler ok, any cross-precision mismatch rejected, fp16 without
  scaler state rejected, v1 (precision-less) checkpoint defaults to fp32.
- **Exact FP32 CPU smoke: train 17.793315887451172, val 18.236953735351562,
  param_hash 1200.7129848422692 — the M3 canonical values to all digits.**
  The AMP changes did not move the FP32 baseline at all.
- Evidence separation: an FP32 summary carries `"precision": "fp32"` and
  `skipped_steps: 0`, `scaler_final_scale: null`, so it can never be
  mistaken for AMP evidence.
- DDP rank-0 writes: two workers launched directly (not via torchrun —
  torchrun's own TCPStore rendezvous hangs in this VM, the known EXP-005
  loopback corruption) produced exactly one SUMMARY line and one
  checkpoint line in merged output; summary reports `ddp: true`,
  `world_size: 2`.

Explicitly UNTESTED here (no GPU — marked, not claimed): actual CUDA AMP
training, autocast numerics, skipped-step behavior under real inf grads,
FP16 tok/s, AMP checkpoint resume on CUDA, and DDP+AMP on GPU. The Kaggle
notebook covers single-GPU AMP training + AMP resume; DDP+AMP measurement
is deferred to a separate run after the single-GPU verdict.

## 4. Checkpoint format and resume rules

- Format stays v1 (no version bump needed: the scaler key is additive and
  old loaders ignore unknown keys). AMP checkpoints add `"scaler"` (state
  dict) and record `"precision"` inside `config["train"]` (argparse
  captures it automatically).
- v1 checkpoints (no precision key, no scaler) load as FP32.
- Resume rules (hard errors, never silent changes):
  - checkpoint precision != requested precision → refuse;
  - FP16 resume from a scaler-less checkpoint → refuse;
  - T_max mismatch → refuse (pre-existing rule, unchanged);
  - missing checkpoint with `--resume` → refuse (pre-existing rule).
- On AMP resume, the log line `resumed scaler scale=X` proves the scaler
  state was actually restored (the notebook asserts it).

## 5. Kaggle benchmark design (notebook `forge_m6_amp_bench.ipynb`)

Single GPU (T4) only. Identical config for both runs; the ONLY difference
is `--precision`:

| setting | value |
|---|---|
| model | 25.3M (d=512, L=8, 8 heads / 4 KV), params_exact 25,305,600 |
| batch × seq | 16 × 512 = 8,192 tokens/step |
| steps / warmup | 300 / 20 (warmup excluded from steady tok/s) |
| optimizer / LR | AdamW, lr 3e-4, wd 0.1, cosine T_max=300 |
| val | at step 300 only (one val pass inside the measured region, identical for both) |
| dirs | `checkpoints/amp_bench/fp32` vs `checkpoints/amp_bench/fp16` |

Plus a fresh-process AMP resume test (20 → 40 steps, T_max=40) and a
cross-precision resume rejection check, both asserted in-notebook.
Output: `/kaggle/working/amp_bench_results.json` (schema in
`amp_bench_results.template.json`).

Timing honesty: steady tok/s is measured over a CUDA-synchronized region
after warmup; overall tok/s is reported for reference only. Val passes
inside the measured region are identical for both precisions, so the
FP16/FP32 ratio is unaffected (documented in the code, not hidden).

## 6. MFU and projection methodology (no numbers yet — NOT RUN)

- FLOPs/token ≈ 6N + 12·L·d·S (N exact, ±10% methodology band).
- MFU = measured steady tok/s × FLOPs/token ÷ peak, where peak matches
  the precision actually used: FP32 runs vs 8.1 TFLOPS, FP16 runs vs
  65 TFLOPS (T4 FP16 tensor core). Comparing across denominators is a
  category error and is not done.
- Scope is per-GPU = aggregate (single-GPU runs). DDP figures, when they
  come, will be reported both ways.
- Speedup = MEASURED steady tok/s ratio. 65/8.1 ≈ 8x is a peak ratio for
  pure matmul, not a training speedup — it is never quoted as one.
- 125M/2.5B projection: scaled from measured AMP steady tok/s by the
  FLOPs/token ratio (125M config at S=512, same assumption as EXP-005).
  25M→125M scaling is approximate (attention fraction, memory behavior,
  long-run effects differ) — the notebook reports a base figure, a
  +15% overheads figure (setup/validation/checkpointing), and a +50%
  planning figure (overheads + a partial failed/repeated run), all labeled
  estimates.

## 7. GPU results — RUN 2026-10-24 (Kaggle, real hardware)

| item | value |
|---|---|
| GPU model / count | 2× Tesla T4, 15,636 MB VRAM each (benchmark runs used 1 GPU) |
| Software | torch 2.11.0+cu128, CUDA 12.8, Python 3.13.15 |
| Commit pin | pinned = actual HEAD = `7d3e58ff` (AMP implementation) |
| FP32 steady tok/s, MFU | 15,204 tok/s, MFU 33.2% (vs 8.1 TFLOPS FP32 peak; steady excludes warmup, hence higher than EXP-005's 30.9% overall) |
| FP16 steady tok/s, MFU | 39,356 tok/s, MFU 10.7% (vs 65 TFLOPS FP16 peak) |
| MEASURED speedup | **2.59x** (steady tok/s ratio; not assumed 8x) |
| FP32 losses | train 0.17259, val 0.17555 — bit-identical to EXP-005 Run B |
| FP16 losses | train 0.16999, val 0.17371 (train/val ratio 1.02; finite throughout) |
| FP16 skipped steps / non-finite counts / scaler scale | 1/300 skipped (0.33%), 0 non-finite losses, 1 non-finite grad event (caught and skipped by design), final scale 32,768 (halved once from 65,536 after the single event, no repeated overflow) |
| FP16 peak memory | 2,554 MB allocated (vs FP32 3,488 MB) |
| AMP resume test | PASS: 20 → 40 steps fresh-process resume, scaler restored, cross-precision resume refused |
| 125M/2.5B projection from measured AMP | 75.3 h base, 86.6 h with 15% overheads, 112.9 h planning (+50% incl. a partial failed run) |

Why FP16 MFU (10.7%) is far below FP32 MFU (33.2%) while still 2.59x
faster: only matmuls run at tensor-core speed; norms, softmax, loss,
optimizer, and casts stay FP32, and at 25M the model is small enough that
kernel-launch and non-matmul overhead dominate (Amdahl effect). At 125M
the matmul fraction rises, so the projection may be pessimistic — but
that is unmeasured and NOT claimed; the projection uses the measured
25M throughput scaled by FLOPs/token ratio, with the scaling caveat.

## 8. Decision: CONDITIONAL PASS

**AMP as a mechanism: PASS.** On the same T4, same config, FP16 trains
2.59x faster than FP32 (39,356 vs 15,204 steady tok/s), losses track FP32
closely (0.170/0.174 vs 0.173/0.176), only 1 of 300 steps skipped, no
repeated overflow, memory down 27%, and fresh-process resume with scaler
restore works (plus cross-precision resume is refused as designed).

**M5 pretraining: still BLOCKED, on a budget decision, not a technical
one.** The projection improved 2.9x (218 h FP32 → 75.3 h FP16 base,
112.9 h planning), but 75–113 h is NOT the original 23–32 h band: it is
roughly 2.5–4 weeks of Kaggle's ~30 GPU-h/week quota, split across
~7–10 sessions of ≤12 h. Resume is proven, so splitting is safe — but
the commitment is real and must be accepted explicitly, not drifted into.

Remaining gates for a pretraining go: (a) Mohnish accepts the 75–113 h
budget (or scopes down: fewer tokens, smaller first model); (b) corpus
decision (the-stack-v2 gated terms need his own HF acceptance, or the
codeparrot-clean fallback); (c) evaluation gates unchanged
(FORGE-Heldout protected; no scores exist yet and none are claimed).

## 9. Gates (unchanged)

- **M5 pretraining: BLOCKED.** (a) Budget decision pending (75–113 h
  accepted, or scope down). (b) Corpus decision still open (the-stack-v2
  gated terms need Mohnish's own HF acceptance, or the codeparrot-clean
  fallback). (c) Evaluation gates: FORGE-Heldout stays protected; no
  scores exist yet and none are claimed.
- No pretraining started; no GPU results fabricated; no unexecuted test
  described as passed. All GPU numbers above are measured and
  independently recomputed.
