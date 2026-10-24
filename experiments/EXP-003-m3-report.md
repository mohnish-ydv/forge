# EXP-003 — M3 Training Feasibility, Checkpoint Recovery & Data Gate

**Date:** 2026-10-24 · **Environment:** Muse VM, Python 3.12, torch 2.14.1+cpu
(2 threads), no GPU
**Baseline:** git `9721191` (M2 conditional pass); no working components
rewritten except where smoke tests produced evidence of bugs (2 fixes,
both documented below).

---

## Gate A — M2 condition audit (condition-by-condition, with evidence)

**A1. Inspection.** Git history: `9721191` (M2) on top of `e9024e6`
(EXP-001 baseline). M2 report, manifest (`data/m2_a/manifest.jsonl`,
10,000 entries), and config (`data/m2_a/config.json`,
fingerprint `4df34774fea80b4f`) all present and read.

**A2. Secrets in retained corpus: NONE.** All 51 secret-flagged records
have `excluded=true` with reason `possible_secret:*`; 0 of the 51 appear
in `data/m2_a/corpus.jsonl` (verified by id join). Hit types: 49 generic
secret assignments, 1 AWS-style, 1 private key. Secret *values* were never
printed to logs — only counts and pattern names.

**A3. License/provenance of retained data: clean.** All 4,764 retained
records carry a known permissive license (apache-2.0: 1795, mit: 1390,
bsd-3-clause: 1342, bsd-2-clause: 137, unlicense: 73, cc0-1.0: 13,
isc: 10, artistic-2.0: 4); 0 unknown/empty. Single provenance set:
`codeparrot/codeparrot-clean @ 35a59fb0` (public, ungated, 2022-10-10).

**A4. Order verified.** `src/data_pipeline.py` lines 353–366:
exact_dedup → near_dedup → decontam.apply → drop excluded →
assign_split. Dedup and decontamination strictly precede split
assignment. Near-dup losers are excluded (never split); hash-based
splitting additionally guarantees identical content → identical split.
Empirical: 0 SHAs in more than one split (M2).

**A5. Split counts 4319/234/211: explained, intentional.** Configured
0.90/0.05/0.05 gives binomial expectations 4287.6/238.2/238.2
(σ≈20.7/15.0/15.0); observed z-scores +1.52/−0.28/−1.81 — ordinary
hash-bucket variance, not a bug. Hash-based splitting trades exact
proportions for dup-proofness and byte-identical reproducibility
(M2: two runs, identical manifests). Intentional; documented here.

**A6. Training-corpus source: still open.** Probed 2026-10-24:
- `HuggingFaceTB/stack-edu` (Python): **metadata-only** (no content
  column); SWH blob fetch → NotFound on sampled IDs.
- `HuggingFaceTB/smolLM-corpus` → `python-edu`: **metadata-only** too.
- `bigcode/the-stack` v1: **gated** (auth required).
- `codeparrot/codeparrot-clean`: **content + license columns, public,
  ungated**, Hub tag 1M–10M rows. Verified working (M2 sample).
  Quality caveat stands: 7.8% unparseable, 41% copyleft in raw sample —
  our pipeline removes both, ~48% yield to clean permissive Python.

Assessment: no verified high-quality content-bearing corpus yet.
codeparrot-clean (filtered) is the fallback v0 corpus; the quality path
(stack-edu content) needs either user-gated the-stack-v2 acceptance or a
proven SWH fetch. **Condition remains OPEN — carried forward.**

**A7. Sample labeling.** `data/README.md` added: the 10k sample is
explicitly a pipeline-validation sample, not a production training
corpus; benchmark material lives under quarantine rules.

---

## Gate B — Training baseline

- **Runtime:** least-expensive legitimate option — official PyTorch CPU
  wheel (`pip install torch --index-url .../whl/cpu`), free, no GPU in
  this VM. torch 2.14.1+cpu. (First install attempt was interrupted by a
  service restart leaving a partial install → diagnosed via missing
  `.so`, clean `--force-reinstall` fixed it.)
- **Implementation:** `src/train.py` — configurable (model + training
  hyperparams via CLI), deterministic toy data (`src/toy_data.py`,
  sha256 `1ea08995…`, byte-level), causal LM loss with shifted targets,
  no padding anywhere (fixed-length chunks, partial batch dropped;
  shape asserted every step), grad clipping (1.0), AdamW + cosine
  schedule, fixed seeds, `torch.use_deterministic_algorithms(True)`,
  per-step logging (loss, tokens, tok/s, elapsed) + validation loss.
- **Bug found by smoke test (fixed with evidence):** `model.py` used
  `.view()` on a non-contiguous target slice → RuntimeError; changed to
  `.reshape()`. Second find: over-long sequences hit RoPE with a
  confusing broadcast error → now raises a clear `ValueError`.
- **Smoke run:** 1.02M-param model, 20 steps: loss 91.31 → 52.44,
  val 48.08; 40 steps: loss → 17.79, val 18.24. **~3,600–4,300 tok/s**
  measured on 2 CPU threads (this number describes the toy setup only;
  NOT used for GPU projections).

## Gate C — Checkpoint recovery (mandatory gate: PROVEN)

Checkpoint v1: model + optimizer + scheduler states, step, tokens_seen,
config, torch+python RNG states. Atomic write (tmp → fsync → os.replace),
validated before rotation; rotation keeps last 2, never deletes the
known-good checkpoint before the new one validates.

- **Kill/resume, bit-identical:** 40 steps uninterrupted vs 20 steps +
  fresh-process resume to 40 → **identical final_train_loss
  (17.793315887451172), final_val_loss, and param_hash
  (1200.7129848422692)**. `deterministic_algos=True`.
- **Confound caught and fixed:** first comparison diverged because
  `CosineAnnealingLR(T_max)` was tied to `--steps` (20 vs 40). Added
  explicit `--t-max` plus a resume-time mismatch guard that refuses to
  resume across different schedules. This is the kind of silent
  experimental error the gate exists to catch.
- **Corruption safety:** `--resume` with no checkpoint → exit 1,
  `FileNotFoundError` (no silent fresh run). Truncated checkpoint →
  exit 1, `RuntimeError: ...zip archive... corrupted`. Both verified.

## Gate D — Training-loop validation: 8/8 PASS

`tests/test_train.py` (torch, 13.4 s): finite loss ✓, shapes agree ✓,
finite grads ✓, params update ✓, checkpoint round-trip reproduces
eval-mode outputs exactly ✓, validation doesn't touch params ✓, loss
decreases on memorizable data ✓, fresh-process subprocess resume
completes ✓.

## Gate E — Compute & size estimates

Measured here: 1.02M model at ~4k tok/s on 2 CPU threads (toy only).

| Model | Params | bf16 ckpt | Train mem (~12 B/param + act) | Q4 GGUF | KV@2k |
|---|---|---|---|---|---|
| 30M | 28.3M | 0.06 GB | ~0.3 GB | ~17 MB | ~8 MB |
| 125M | 124.1M | 0.25 GB | ~1.5 GB | ~73 MB | ~28 MB |
| 350M | 349.2M | 0.70 GB | ~4.2 GB | ~206 MB | ~59 MB |

GPU-hours are **derived, not measured** (no GPU in this VM):
FLOPs=6ND ÷ (T4 65 TFLOPS × MFU 25–35%). Batch 8×2048 assumed.

| Target | Tokens | T4-hours (range) | Steps (~) |
|---|---|---|---|
| 30M smoke | 0.5B | 1.0–1.5 | ~30k |
| 125M v0 | 2.5B | 23–32 | ~153k |
| 350M probe | 1.0B | 26–36 | ~61k |
| 350M Chinchilla | 7.0B | 179–251 | ~427k |

Uncertainty: MFU on free tiers is the dominant unknown (±30%);
Kaggle queueing/wall-clock quota not modeled. **A representative GPU
smoke run on Kaggle is required before trusting these numbers.**
Phone inference: separate deployment measurement, not performed (no
device); sizes above are arithmetic from verified param counts.

## Gate F — Decision: CONDITIONAL PASS

- [x] Tiny model trains; loss decreases (91.3 → 17.8 over 40 steps)
- [x] All automated tests pass (26 pipeline + 6 decontam + 8 train + harness)
- [x] Checkpoint recovery demonstrated with bit-identical evidence;
      corruption fails safe; rotation preserves known-good
- [x] M2 conditions audited: secrets 0 in corpus, licenses clean,
      order verified, splits explained
- [ ] **Training-corpus source: OPEN** (carried from M2) — no serious
      pretraining or substantial GPU allocation until resolved

**Next:** M4 (finalize eval harness + FORGE-Heldout draft) can proceed;
M5 pretraining stays blocked on the corpus decision. The two realistic
paths are user-gated the-stack-v2 acceptance (quality) or the
codeparrot-clean filtered subset (fallback, ~48% yield).

## Artifacts

- `src/train.py`, `src/toy_data.py`, `tests/test_train.py`
- `src/model.py` (2 correctness fixes), `data/README.md`
- `checkpoints/gate_c2/` (bit-identical A/B runs), `checkpoints/smoke_test/`
- This report: `experiments/EXP-003-m3-report.md`
- Repro commands: `.venv/bin/python src/train.py --steps 40 --t-max 40
  --ckpt-every 10 --ckpt-dir checkpoints/gate_c2/run_a`
