# EXP-007 — M5 readiness: corpus audit + larger-model pilot

## Status: AUDIT COMPLETE, PILOT PREPARED (NOT RUN), FULL RUN NOT APPROVED

AMP conditionally passed in EXP-006. This experiment answers the two
questions that blocked M5 after EXP-006: (1) is codeparrot-clean a real,
licensable, large-enough corpus — measured, not assumed? (2) does the
125M-class model actually train at the projected throughput on a real
code slice, with kill/resume proven at that scale?

**The full 125M / 2.5B-token pretraining run does NOT start from this
report.** It needs a separate approval after the pilot evidence below is
reviewed. The recommendation at the end of this version is: run the
pilot first.

## 1. Baseline freeze (verified 2026-10-24)

| item | value |
|---|---|
| GitHub head | `0fee1758` — "FORGE: EXP-006 AMP results + verdict" |
| Local head | `aca274d` — independent AMP benchmark analyzer (to push with this report) |
| AMP implementation commit | `7d3e58ff` (GitHub snapshot; `src/train.py` / `src/model.py` unchanged since) |
| EXP-006 decision | CONDITIONAL PASS, confirmed independently: analyzer run on the archived `amp_bench_results.json` gives 38 ok, 2 warnings, 0 failures, verdict CONDITIONAL (planning 112.9 h outside the feasibility band) |
| AMP protections re-verified | `tests/test_amp.py` 14/14 green, fresh run (precision validation, scaler save/restore, resume rejection rules, DDP rank-0 writes, exact CPU FP32 smoke) |
| FP32 / FP16 evidence intact | `experiments/amp_bench_results.json`: FP32 steady 15,204 tok/s, FP16 steady 39,356 tok/s, status `ok` — unchanged |
| Frozen training configuration (both precisions) | 25.3M model, 300 steps, batch 16×512, warmup 20, AdamW lr 3e-4 wd 0.1, cosine T_max=300, seed 42 — identical except `--precision` |

No code change since EXP-006 touches the training path, so the AMP
checkpoint-recovery and configuration-mismatch protections verified in
EXP-006 remain active by construction and by the fresh test run above.

## 2. Corpus audit — codeparrot-clean

### 2.1 Source, revision, provenance

- **Source:** `codeparrot/codeparrot-clean` on HuggingFace Hub.
- **Revision pinned:** `35a59fb025bc0a102f7d96eac09d145b896d487b` (resolves
  on the Hub; the M2 sample's first record matches shard 1's first record
  byte-for-byte, proving the sample and the audit slice come from the
  same revision and the same row order).
- **Inventory:** 54 shards, 12.81 GB compressed (shards 1–53 ≈ 230–247 MB
  each, shard 54 ≈ 142 MB). Shard 1 contains exactly 100,000 rows.
- **Language:** Python only — codeparrot-clean is a Python-only dataset;
  the pipeline's language filter is `python`.
- **Dataset-level license:** NONE. The Hub card carries no license field
  (cardData: tags only). All licensing rests on the per-file `license`
  field. The M2 acquisition note records: dataset code Apache-2.0,
  underlying files per their original repositories.

### 2.2 Measured sample: first 10,000 rows (M2, re-verified)

Pipeline: prefilter → permissive-license filter → quality (parse,
alpha-frac) → secrets scan → exact dedup (SHA-256) → near-dup (5-gram
Jaccard ≥ 0.7) → HumanEval+ decontamination → hash split. Re-verified
from `data/m2_a/manifest.jsonl` (not from memory):

| stage | count |
|---|---|
| discovered | 10,000 |
| after prefilter (generated removed: 222) | 9,778 |
| after license filter (copyleft/unknown removed: 4,039) | 5,739 |
| after quality (unparseable 776, low-alpha 10) | 4,953 |
| after secrets (flagged 51) | 4,902 |
| after exact dedup | 4,902 |
| after near-dup (138 removed, 279 candidate pairs) | 4,764 |
| **retained** | **4,764 (47.6%), 44,826,171 bytes, ~11.2M tokens (bytes/4 estimate)** |
| splits | train 4,319 / val 234 / test 211 |
| file sizes (kept) | mean 9,409 B, median 4,366 B, max 749,187 B |
| retained with secret hits | **0** |
| kept SHAs in more than one split | **0** |
| heldout task hits (prompt+tests, M4 check) | **0** |

License distribution, all 10,000 rows (kept in bold): apache-2.0 2,150
(**1,795**), mit 1,801 (**1,390**), bsd-3-clause 1,655 (**1,342**),
bsd-2-clause 164 (**137**), unlicense 82 (**73**), cc0-1.0 16 (**13**),
isc 14 (**10**), artistic-2.0 6 (**4**) — permissive total kept 4,764.
Excluded at the license stage: gpl-3.0 1,730, gpl-2.0 949, agpl-3.0 917,
mpl-2.0 159, lgpl-3.0 146, lgpl-2.1 121, epl-1.0 17 (stage counts differ
slightly from raw counts because prefilter runs first). Raw mix: ~41%
copyleft. Secret exclusions: 49 generic-secret assignments, 1 AWS secret
assignment, 1 private key.

### 2.3 Audit slice: shard 1, 100,000 rows (10x M2) — STAGED

Run with `experiments/audit_shard1.py` (stage code validated: it
reproduces the M2 10k stats EXACTLY — all counts identical, excluding
elapsed time). The audit is staged because the naive full pipeline does
not scale on this VM, which is itself a finding (see below):

- **Stage A (100k rows, all stages except near-dup):** prefilter →
  license → quality → secrets → exact dedup → HumanEval+ decontamination
  → hash split. Writes `data/audit_m2/` (git-ignored; only the numbers
  below enter this report).
- **Stage B (45k rows, full pipeline including near-dup):** measures the
  near-dup rate at pilot scale and the near-dup stage's memory. 45k rows
  is the pilot's corpus size, so Stage B also proves the pilot's
  corpus-prep fits in RAM.
- Checks (both stages): heldout-task isolation (M4 check, extended),
  split isolation (SHA in >1 split), secret-kept = 0, manifest hygiene
  (hashes/paths/licenses only — `write_manifest` excludes `content` by
  construction; verified in code).

**Scaling finding (measured):** near-dup on the 10k validation run took
RSS from 171 MB to 1,019 MB for 4,902 survivors (~170 KB per survivor in
shingle sets). At 50k survivors that projects to ~8.5 GB — beyond this
VM's 7.9 GB — and the two naive 100k audit attempts died mid-run with
RSS climbing past 5 GB. Consequence: the current near-dup implementation
cannot build the full 5.2M-file corpus on ordinary hardware. The
full-corpus build needs a sharded or MinHash/LSH dedup design (deferred
design task, NOT blocking the pilot, which is 45k rows). The 100k audit
therefore reports retention *before* near-dup, plus the near-dup rate
measured at 45k, and the full-corpus token estimate in §2.4 is adjusted
by that rate.

Results below are measured (Stage A + Stage B), not estimated.

| item | value |
|---|---|
| rows in | Stage A: 100,000 (shard 1, all stages except near-dup). Stage B: 45,000 (shard 1, full pipeline incl. near-dup) |
| retained (rate) | A: 49,135 (49.1%, before near-dup). B: 20,722 (46.0%, full pipeline). Consistency check: 49.1% × (1 − 6.5% near-dup rate) = 45.9% ≈ B's direct 46.0% |
| bytes retained / tokens est (bytes/4) | A: 456,357,829 B / 114.1M tokens. B: 185,605,107 B / 46.4M tokens |
| license mix (kept, Stage A) | apache-2.0 19,161; mit 14,144; bsd-3-clause 13,356; bsd-2-clause 1,410; unlicense 756; isc 134; cc0-1.0 128; artistic-2.0 46 (sums to 49,135) |
| exclusion breakdown (Stage A) | prefilter 2,255 (generated 2,247, binary 1, minified 7); license 40,237; quality 7,736 (unparseable 7,632, low-alpha 104); secrets 637; exact dedup 0; decontamination 0. Every stage reconciles exactly (in − out = excluded) |
| exclusion breakdown (Stage B) | prefilter 992 (generated 991, binary 1); license 18,130; quality 3,439 (unparseable 3,388, low-alpha 51); secrets 268; exact dedup 0; near-dup 1,449 (1,844 candidate pairs); decontamination 0 |
| near-dup rate | 6.5% of post-exact survivors at 45k (1,449/22,171), vs 2.8% at 10k (138/4,902) — rises with scale, as expected: more docs mean more chances to find a near-duplicate |
| file sizes (kept) | A: mean 9,288 B, median 4,221 B. B: mean 8,958 B |
| splits | A: train 44,163 / val 2,515 / test 2,457. B: train 18,668 / val 1,052 / test 1,002 (≈90/5/5 by hash, as designed) |
| retained with secret hits | 0 (both stages) |
| kept SHAs in more than one split | 0 (both stages) |
| heldout task hits | [] (both stages) |
| manifest carries content? | No — verified: manifest keys are bytes/excluded/exclusion_reason/id/language/license/path/provenance/repo/secret_hits/sha256/source/source_rev/split only |
| Stage A elapsed / peak RSS | 397.7 s / 1,338 MB (no near-dup) |
| Stage B elapsed / peak RSS | 301.3 s / 5,658 MB (with near-dup — the memory cost that killed the naive 100k attempts) |

### 2.4 Full-corpus extrapolation (estimate, method stated)

Shard 1 is exactly 100,000 rows at 246.1 MB compressed. Total compressed
12.81 GB ÷ 246.1 MB ≈ 52.05 shard-equivalents → **≈ 5.2M rows total**
(estimate: per-shard row density varies; shard 54 is smaller). At the
measured full-pipeline retention rate (46.0%, Stage B direct) and mean
kept size (8,958 B, Stage B):

- ≈ 2.39M retained files, ≈ 21.4 GB, ≈ **5.4B tokens (bytes/4 estimate)**.
- That EXCEEDS the 2.5B-token plan by more than 2x: on these rates,
  codeparrot-clean alone can supply the full run's unique tokens — no
  the-stack-v2 dependency for volume.

Caveats on the extrapolation (stated, not buried): (1) near-dup removal
is measured within slices (6.5% at 45k) and applied to the 100k figure
above, but cross-shard exact/near duplicates are NOT removed by a
per-shard audit, so true unique tokens are somewhat below 5.4B — the
full-corpus build must run global exact-dedup (cheap: SHA set) and
report the post-dedup count before the full run; (2) the bytes/4 token
estimate is a heuristic — the pilot measures `tokens_seen` exactly and
the full run's data build will count tokenized tokens; (3) shard 1 may
not be perfectly representative (early shards skew to particular repos
— the pilot's shard-2 slice is a second sample that tests this).

### 2.5 Licensing decision: ACCEPTED (conditional, uncertainties flagged)

**Decision:** the codeparrot-clean *permissive subset* (MIT, Apache-2.0,
BSD-2/3-Clause, ISC, Unlicense, CC0-1.0, Artistic-2.0, as implemented in
`PERMISSIVE_LICENSES`) is accepted as the v0 corpus for the pilot and,
subject to the pilot, for the full run. Copyleft-licensed files
(GPL/LGPL/AGPL/MPL/EPL, ~41% of raw rows) stay excluded — the conservative
choice, kept deliberately.

**Working interpretation (not legal advice):** permissive licenses do
not attach copyleft obligations to model weights trained on the code;
Apache-2.0's patent grant covers the code itself. Nothing in the
retained set requires the trained weights to be distributed under a
copyleft license.

**Uncertainties flagged, not assumed away:**
- License metadata is self-reported per file at collection time and was
  not independently verified per repo. A wrongly-tagged file can slip
  through; the filter is only as good as the metadata.
- There is no dataset-level license to fall back on — if per-file
  metadata is wrong, there is no second layer.
- Reworded/paraphrased copyleft code escapes verbatim/near-verbatim
  filters (known blind spot since M2). FORGE-Heldout remains the
  backstop for benchmark integrity, not for license integrity.
- This section is an engineering assessment, not a legal opinion. If
  FORGE is ever distributed commercially, get a real license review
  before that distribution.

The-stack-v2 remains a deferred alternative (its gated terms still need
Mohnish's own HuggingFace acceptance) — it is no longer needed for
volume, only if the full-corpus measurement contradicts §2.4.

### 2.6 What the audit does NOT claim

- No secret values are printed anywhere in this report or in any
  manifest (manifests carry hashes, paths, licenses, and exclusion
  reasons only).
- No quarantined heldout solutions are included in any training manifest
  (the heldout check uses task prompts+tests only, never solutions).
- The 10,000-file M2 sample remains labelled pipeline-validation-only;
  the audit slice and the pilot slice are the first corpus measurements
  at scale.

## 3. Larger-model pilot (prepared, NOT RUN)

Notebook: `notebooks/forge_m7_pilot.ipynb` (pin filled after push; all
code cells compile; corpus-prep cell logic simulated locally on 200 rows:
112 retained, 0 secret-kept, 0 heldout hits, train.bin built correctly).

| item | design |
|---|---|
| Model | 125M-class: d=768, L=18, 12 heads / 3 KV heads, byte vocab 256 → 111,701,760 params (same config the EXP-006 projection uses) |
| Precision / device | FP16 AMP, single T4 (session may be T4 x2; the benchmark uses one GPU) |
| Data | Shard 2 (not the audit's shard 1), first 45,000 rows → same pipeline → `data/pilot/train.bin`. Expected at audit rates: ~21,000 files, ~200 MB, ~50M tokens — ~1 epoch at 6,100 steps × 8,192 tokens |
| Phase A probe | 300 steps, T_max=6100 (full schedule), warmup 20, ckpt-every 300 — measures 125M steady tok/s + peak memory directly |
| Kill/resume test | Separate dir, fresh 200-step run, ckpt-every 50, killed mid-run (~75 s), fresh-process `--resume` to 200, scaler restore asserted — the mechanism a multi-session full run depends on, tested at full model size |
| Phase B main run | `--resume` to 6,100 steps, ckpt-every 1000, val-every 1000, log-every 100, full log kept for the loss curve |
| Resume cell | Guarded: skips itself if the summary already shows 6,100 steps (re-running a completed run would crash on an empty step range); otherwise resumes from the latest checkpoint |
| Eval | Dev split ONLY (12 tasks) — held-out split never touched. Greedy generation (max_new 512, truncate at first new top-level `def`), harness `evaluate`, plus empty and `return None` baselines re-run on dev through the identical driver |
| Results JSON | `/kaggle/working/pilot_results.json`: hardware, commit, corpus stats, probe + main summaries, kill-test outcome, loss curve sample, dev eval + baselines, scaling comparison (measured 125M vs EXP-006 25M vs same-MFU prediction), updated projection from MEASURED 125M tok/s |
| GPU budget | Probe ~5 min + kill test ~5 min + main run ~60–90 min + eval ~15 min ≈ **under 2 hours** — inside one 12 h session and a small fraction of the ~30 GPU-h/week quota |

**Expected pilot outcome (stated before the run):** dev score near zero
is EXPECTED at 50M tokens (~2% of the plan) and validates the pipeline —
it is not a capability claim in either direction. A pilot FAIL would be:
AMP instability at 125M (repeated overflows, most steps skipped), OOM at
batch 16×512, kill/resume failure, or throughput so low the projection
leaves the feasible band entirely.

**Pilot status: NOT RUN.** No pilot numbers are claimed in this version.
Section 4's capability evaluation awaits the pilot checkpoint.

## 4. Evaluation and safety gates (M4, re-verified)

- `tests/test_heldout.py`: **21/21 green** (fresh run 2026-10-24) —
  heldout frozen (24 tasks, sha `a156b8bd081e7315…`), oracles 24/24,
  baselines empty 0/24 and `return None` 0/24, contamination 0/24 vs
  HumanEval+ and vs the M2 corpus, sandbox mode recorded per result.
- Eval harness: untrusted model output runs only in isolation (pivot
  userns sandbox where available; plain-subprocess fallback is weaker
  and its mode is recorded per result). The pilot eval runs on Kaggle,
  where userns availability is unknown — the harness records the mode
  actually used, and the pilot JSON must be checked for it.
- Toy-data loss is never coding capability. No FORGE benchmark scores
  exist yet and none are claimed in this report.

## 5. Updated compute estimate (with uncertainty)

EXP-006 projection (FLOPs-ratio from measured 25M FP16 throughput):
**75.3 h base, 86.6 h with 15% overheads, 112.9 h planning** for
125M / 2.5B tokens. That projection assumes 125M MFU = 25M MFU — an
assumption the pilot replaces with a measurement.

Sensitivity to that assumption (before the pilot):

| 125M MFU vs 25M MFU | base hours | planning hours (×1.5) |
|---|---|---|
| 0.8x (worse at scale) | 94.1 | 141.2 |
| 1.0x (EXP-006 assumption) | 75.3 | 112.9 |
| 1.2x (better at scale) | 62.8 | 94.1 |

The pilot's probe steady tok/s converts this table into a single
measured number (hours = 2.5B ÷ measured tok/s). Until then, the honest
band is **~63–94 h base, ~94–141 h planning** — all multi-week on
~30 GPU-h/week, all splittable across sessions because resume is proven
(M3 on CPU, M6 on GPU, pilot kill-test at 125M scale pending).

Pilot cost itself: under 2 GPU-hours, one session, no quota risk.

## 6. Outstanding blockers

1. **Pilot not run** (Kaggle) — the main blocker. Everything downstream
   (measured 125M throughput, real-data stability, pipeline validation)
   waits on it.
2. Full-corpus unique-token count after global exact-dedup — estimated
   only (§2.4) until the full-corpus build runs.
3. **Near-dup at full-corpus scale needs a redesign** (sharded or
   MinHash/LSH) — the current implementation OOMs past ~50k survivors on
   8 GB RAM (§2.3). Not blocking the pilot (45k rows), blocking the
   full-corpus build.
4. Budget decision: accept the pilot-revised 125M budget (or scope down:
   fewer tokens, smaller first model) — a human decision, not a technical
   one.
5. Corpus condition: the §2.5 acceptance is conditional on the pilot's
   shard-2 slice behaving like the audit slice (retention, licenses,
   secrets, heldout isolation all re-checked in the notebook).
6. the-stack-v2: deferred alternative, not required for volume.

## 7. Recommendation: RUN THE PILOT FIRST

- **Proceed to the full run? No** — not before measured 125M throughput
  and real-data stability exist. A 75–113 h commitment on an extrapolated
  number would repeat the EXP-003→EXP-005 error (assumed 65 TFLOPS FP16
  vs measured FP32 reality).
- **Stop and redesign? No** — nothing has failed. AMP is proven, the
  corpus measures viable on two samples, the pipeline is validated, and
  the pilot is cheap (<2 h) and de-risks exactly the two unknowns that
  matter.
- **Run another pilot? This IS the pilot.** Run `forge_m7_pilot.ipynb`
  on Kaggle (GPU), download `pilot_results.json` + `corpus_stats.json`,
  and return them. EXP-007 v2 then fills in the pilot results and issues
  the final recommendation: proceed / another pilot / stop and redesign,
  with the updated compute commitment stated plainly for approval.

No full pretraining starts automatically. The next approval requested is
for the pilot run itself (already designed) and, after its evidence, for
the full run's budget.
