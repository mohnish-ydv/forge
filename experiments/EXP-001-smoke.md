# EXP-001 — Pipeline smoke test (CPU, no training yet)

**Date:** 2026-10-24 · **Environment:** Muse VM, Python 3.12.3, no torch, no GPU
**Hypothesis:** Model config arithmetic, BPE tokenizer, and eval harness
work end-to-end on CPU before any GPU spend.
**Baseline:** B0 floor — empty body must score 0; oracle must score 1.0.

## Results (verified by running the code)

### Model param arithmetic (`python3 src/model.py`)
- FORGE-30M: 28.32M total, Q4 ~17 MB, KV@2k 8 MB
- FORGE-125M: 124.09M total, Q4 ~73 MB, KV@2k 28 MB
- FORGE-350M: 349.23M total, Q4 ~206 MB, KV@2k 59 MB
- First draft configs under-counted (87M/304M); layer counts corrected
  to 18/28 so labels match arithmetic. Lesson: never trust a label,
  always run the counter.

### Tokenizer (`python3 src/tokenizer.py`)
- Byte-BPE trained to vocab 470 on 6 sample functions (x20 repeats)
- Lossless round-trip: PASS on train + held-out samples
- Fertility on held-out code: 0.536 tok/byte (1.87 bytes/token) vs
  1.0 for raw bytes — BPE helps even at tiny scale
- Fertility on train data is degenerate (whole samples merged into
  single tokens) — confirms fertility MUST be measured held-out.

### Eval harness (`python3 benchmarks/eval_harness.py`)
- Empty (`pass`) completions: pass@1 = 0.0, funnel = 3x assertion ✓
- Oracle completions: pass@1 = 1.0 ✓
- Broken completions classified correctly: syntax / assertion / runtime ✓

## Decision
Pipeline code is sound. Next: M2 data-pipeline sample + M3 smoke
training (needs torch — install CPU wheel locally, CUDA on Kaggle).
Do NOT start 125M pretraining until kill-resume checkpoint test passes.
