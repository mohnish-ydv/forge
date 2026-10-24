# EXP-002 — M2 Data Pipeline, Leakage Audit & Reproducibility Gate

**Date:** 2026-10-24 · **Environment:** Muse VM, Python 3.12.3 stdlib (+venv with
`datasets`/`evalplus` for acquisition/fingerprints only), no GPU
**Baseline:** git `e9024e6` (EXP-001 artifacts); all EXP-001 tests re-ran green.

---

## Step 1 — Artifact inspection (findings)

- All 9 EXP-001 files present and re-executed: `model.py`, `tokenizer.py`,
  `eval_harness.py` all pass their self-tests (param counts verified,
  tokenizer lossless, harness B0=0.0 / oracle=1.0).
- **Discrepancy 1 (fixed):** no git repo existed despite the roadmap
  requiring version control → initialized, baseline committed as `e9024e6`.
- **Discrepancy 2 (fixed):** M1 promised unit tests; only `__main__`
  self-tests existed → added `tests/test_pipeline.py` (26 tests) and
  `tests/test_decontam.py` (6 tests), all passing.

## Step 2 — Pipeline (`src/data_pipeline.py`, pure stdlib)

Stages: prefilter (language/size/empty/binary/generated/minified) →
license (permissive-only, configurable) → quality (unparseable, alpha-frac)
→ secrets scan → exact dedup (sha256) → near-dup (inverted-index candidates
+ exact Jaccard ≥ 0.7) → decontamination hook → hash-based train/val/test
split → JSONL manifest (no content field) + corpus.

Config fingerprint: `4df34774fea80b4f` (recorded in stats.json).

## Step 3 — Acquisition (bounded sample, full provenance)

- Source: `codeparrot/codeparrot-clean` @ `35a59fb025bc0a102f7d96eac09d145b896d487b`
  (public, ungated, 2022-10-10) — `data/provenance.json`
- Method: first 10,000 rows of fixed streaming order (deterministic)
- Acquired 2026-10-24T02:58:34Z, 103 MB raw JSONL
- **Why not stack-edu:** verified metadata-only on the Hub (no content
  column; README confirms); SWH blob fetch returned NotFound for sampled
  IDs; `bigcode/the-stack` v1 is now gated. Documented as the open
  acquisition problem for M5 — this sample validates the *pipeline*, not
  the final corpus.

## Step 4 — Pipeline run results (10,000 files)

| Stage | Kept | Excluded (reason: count) |
|---|---|---|
| discovered | 10,000 | — |
| prefilter | 9,778 | generated: 222 |
| license | 5,739 | license (copyleft/unknown): 4,039 |
| quality | 4,953 | unparseable: 776, low_alpha_frac: 10 |
| secrets | 4,902 | possible_secret: 51 (49 generic assign, 1 aws, 1 private key) |
| exact dedup | 4,902 | 0 (source was pre-deduped) |
| near-dup | 4,764 | 138 (from 279 candidate pairs ≥ 0.7) |
| decontam | 4,764 | benchmark_overlap: 0 |

- **Retained: 4,764 docs (47.6%), 44.8 MB, 1,221,391 lines, ~11.2M tokens
  (est. @4 B/tok)** — Python only
- License mix of raw sample: apache-2.0 2150, mit 1801, bsd-3-clause 1655,
  gpl-3.0 1756, gpl-2.0 974, agpl-3.0 928, … → 41% copyleft correctly
  excluded for the clean release
- Split: train 4,319 / val 234 / test 211 (hash-based)
- Perf: 66 s wall, 64 s CPU, **peak RSS 1,019 MB** (measured via wait4)
- Scaling estimate: ~6.5 ms/file, ~100 MB RAM per 1k files → 1M files
  ≈ 1.8 h single-core, ~10 GB RAM (shardable; near-dup index is the
  dominant cost)

## Step 5 — Contamination audit (HumanEval+ 164 fingerprints)

- Fingerprint source: `evalplus` package v0.1.10 (164 problems,
  prompt+canonical), built by `benchmarks/build_fingerprints.py`,
  quarantined per `benchmarks/quarantine/README.md`
- **Result: 0 overlaps** across L1–L5 on all 4,764 retained docs
- **Positive controls (detectors proven working):**
  - verbatim canonical → L1+L2+L3+L4 fire
  - benchmark code embedded in a larger file → L3+L4 fire
  - unrelated code → clean
- **Documented limitation (demonstrated, not hypothesized):** a reworded
  implementation of the same algorithm produced **zero hits** — n-gram
  methods cannot catch paraphrased contamination. "Zero hits" means
  "zero hits against this 164-doc fingerprint set", never "provably clean".
  FORGE-Heldout remains the backstop.

## Step 6 — Harness security audit (empirical)

Adversarial tests (`benchmarks/test_harness_adversarial.py`), before/after:

| Attack | Before | After mitigation |
|---|---|---|
| read env secrets | HOLE | **blocked** (env cleared) |
| 2 GiB memory hog | unlimited | **limited** (1 GiB rlimit → MemoryError) |
| infinite loop | timeout ✓ | timeout ✓ |
| read file outside tempdir | HOLE | HOLE (residual) |
| write outside tempdir | HOLE | HOLE (residual) |
| spawn subprocess | HOLE | HOLE (residual) |

Conclusion: subprocess+tempdir+timeout+rlimits is **process isolation, not
a security sandbox**. Residual filesystem/subprocess/network holes cannot
be closed with stdlib alone. **Policy: evaluation of untrusted model
output runs only in a throwaway container/VM**, never on machines holding
credentials. Functional scoring (B0=0.0, oracle=1.0) unaffected.

## Step 7 — Reproducibility & split integrity

- Ran pipeline twice (m2_a, m2_b): **manifests byte-identical,
  corpora byte-identical, stats identical** (excluding wall time)
- Split integrity: 4,764 unique SHAs, **0 SHAs in more than one split**
  (hash-based split is dup-proof by construction)

## Acceptance criteria check

- [x] Existing tests pass (EXP-001 self-tests + 26 pipeline + 6 decontam)
- [x] New pipeline tests pass
- [x] No exact duplicate crosses splits (0, verified)
- [x] Repeated runs reproducible (byte-identical)
- [x] License + provenance recorded per retained doc
- [x] Benchmark overlaps reported per policy (0 hits; detectors proven
      with positive controls; limitations documented)
- [x] No unresolved critical data-integrity issue in the M2 sample

## Decision: CONDITIONAL PASS

Proceed to M3 (checkpointed smoke training) and M4 (eval harness
finalization) **with these explicit conditions**:

1. **Training-corpus acquisition is still open.** stack-edu is
   metadata-only; SWH fetch failed on sampled blobs; the-stack v1 is
   gated. M5 pretraining MUST NOT start until a content-bearing,
   license-clear source is validated end-to-end (candidate: re-test
   `HuggingFaceTB/smolLM-corpus` python-edu path, or licensed clone
   set with per-file license records).
2. **Harness runs in throwaway isolation only** for untrusted candidates
   (residual fs/subprocess holes documented above).
3. **Decontamination is verbatim/near-verbatim only** — reworded
   contamination is a known blind spot; FORGE-Heldout is the backstop.

## Artifact paths

- `src/data_pipeline.py`, `src/decontam.py`, `src/acquire_sample.py`,
  `src/run_m2.py`
- `benchmarks/build_fingerprints.py`, `benchmarks/fingerprints.json`,
  `benchmarks/quarantine/README.md`,
  `benchmarks/test_harness_adversarial.py`
- `tests/test_pipeline.py`, `tests/test_decontam.py`
- `data/provenance.json`, `data/raw_sample.jsonl` (git-ignored),
  `data/m2_a/{manifest,corpus,stats,config}.json{l,}`
- This report: `experiments/EXP-002-m2-report.md`
