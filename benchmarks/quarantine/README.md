# Quarantine — benchmark reference data

This directory holds benchmark REFERENCE material used ONLY for:

1. **Decontamination** — checking training data for benchmark overlap
   (see `src/decontam.py` and `benchmarks/fingerprints.json`).
2. **Evaluation** — running the eval harness against reference solutions.

## Hard rules

- Nothing in this directory is ever training data.
- Reference solutions are never used for model tuning, checkpoint
  selection, or prompt engineering of the model.
- The data pipeline's source paths MUST NOT include this directory.
  (Enforced by convention + the M2 pipeline only reading `data/raw_*.jsonl`.)
- `fingerprints.json` (one level up) contains derived fingerprints, so
  the training pipeline never needs the raw solutions at all.

## FORGE-Heldout v1 (added M4)

- `forge_heldout_v1.json` holds the FROZEN benchmark: 12 dev + 12 held-out
  tasks, prompts, hidden tests, per-task SHA-256 hashes, benchmark hash
  `a156b8bd081e7315…`. Any edit to prompts/tests requires a version bump.
- `build_heldout.py` contains the reference (oracle) solutions. It is
  quarantined with this directory: never training data, never model-facing.
- Held-out split rules: the 12 `heldout/*` tasks are for FINAL evaluation
  only. Never train on them, never tune prompts/checkpoints against them,
  never iterate on failures. The 12 `dev/*` tasks may be used for iteration.
- Contamination audit (M4, re-run before any training corpus freeze):
  0/24 tasks hit HumanEval+ fingerprints (L1–L5); 0/24 hit the M2 corpus
  (exact + normalized). Positive control: verbatim HumanEval doc fires
  L1–L4. Known blind spot: reworded/paraphrased contamination is NOT
  caught by n-gram methods — original authorship is the mitigation.

Violating these rules invalidates every benchmark number FORGE reports.
