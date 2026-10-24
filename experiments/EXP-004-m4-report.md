# EXP-004 — M4: Evaluation Integrity, Held-Out Benchmark & Execution Safety

Date: 2026-10-24. Repo: ~/workspace/forge. Parent commit: 60887ad (M3).

## Verdict: CONDITIONAL PASS

Evaluation correctness and benchmark integrity are established with evidence.
Execution isolation is substantially upgraded (genuine userns + pivot_root
sandbox on this machine) but remains platform-dependent — the fallback and
its limits are documented, so Gate C is a conditional pass, not a full pass.

## Gate A — Evaluator audit (PASS)

`benchmarks/eval_harness.py` was audited end to end. One real scoring hole
was found and fixed:

- **Hole (found by adversarial review, proven by execution):** the completion
  marker was a fixed string `__FORGE_PASS__`. A candidate printing that string
  and calling `sys.exit(0)` before the tests ran scored a false PASS
  (demonstrated pre-fix).
- **Fix:** per-run 128-bit token (`secrets.token_hex(16)`); a pass requires
  returncode 0 AND the token in stdout. Post-fix the spoof scores FAIL.
- Verified paths: empty completion -> fail; wrong answer -> `assertion`;
  exception -> `import/runtime`; `while True` -> `timeout`; malformed code ->
  `syntax` (pre-execution `ast.parse` gate); oracle -> pass. Failed tests
  cannot accidentally score: any assert failure is a nonzero exit, and the
  token is unguessable per run.
- B0 preserved: empty-body baseline scores 0.0 on the harness mini-problems
  and 0/24 on FORGE-Heldout.

## Gate B — FORGE-Heldout v1 (PASS)

`benchmarks/build_heldout.py` authors 24 ORIGINAL tasks (12 dev + 12 held-out),
verifies each against a reference solution, and freezes
`benchmarks/forge_heldout_v1.json` (version `forge-heldout-v1`,
benchmark_sha256 `a156b8bd081e7315…`, per-task prompt/test SHA-256, CC0-1.0,
`frozen: true`; any prompt/test edit requires a version bump).

- Oracle (reference solutions, benchmark-correctness check only — NOT a model
  result): 24/24 pass.
- Contamination audit, all with the M2 detector suite:
  - 0/24 tasks hit HumanEval+ fingerprints at any level (L1–L5).
  - 0/24 tasks hit the M2 retained corpus (exact + normalized SHA-256).
  - Positive control: a verbatim HumanEval+ doc fires L1/L2/L3/L4 (detectors
    proven live, not assumed).
- Limitations (documented, unchanged from M2): n-gram detectors cannot catch
  reworded/paraphrased contamination; mitigation is original authorship plus
  the held-out split. Hidden tests and reference solutions are quarantined
  (`benchmarks/quarantine/README.md`, `build_heldout.py` header): never
  training data, never model-facing, never tuned on.

## Gate C — Execution safety (CONDITIONAL PASS)

Threat model: candidate code is untrusted. What was built and measured:

- **New: pivot_root sandbox** (`_pivot_available()` probe, used automatically).
  Each candidate runs under `unshare -rmn` (private user, mount, network
  namespaces), then `pivot_root` into a tmpfs carrying read-only `/usr`,
  `/lib` (`/lib64`), a fresh tmpfs `/tmp` and workspace, `/dev/null`+`/dev/urandom`,
  with the old root masked by a second tmpfs. Verified by execution:
  - reads of `/home/hatch/USER.md` and `/oldroot/...` -> FileNotFoundError
    (host filesystem unreachable);
  - writes to `/`, `/tmp` land in the sandbox tmpfs; host check confirms no
    escape files;
  - network -> `OSError: Network is unreachable`;
  - environment: cleared (`FORGE_TEST_SECRET` canary not visible);
  - fork bomb -> `BlockingIOError` via RLIMIT_NPROC=64 (new);
  - 2 GiB hog -> killed via RLIMIT_AS=1 GiB; infinite loop -> wall-clock
    timeout kills the whole process group (`start_new_session` + `killpg`,
    so forked grandchildren die too); RLIMIT_FSIZE=100 MiB added.
- Adversarial re-audit (`benchmarks/test_harness_adversarial.py`): memory
  LIMITED, env blocked, fs read/write **blocked** (were HOLES in M2),
  subprocess spawn still "reaches" — but children are contained inside the
  same namespaces (no network, no host fs). Downgraded from hole to
  contained-and-documented.
- **Residuals (honest):** the sandbox needs Linux user namespaces
  (probed at runtime; falls back to `sandbox="plain"` subprocess isolation
  where unavailable, e.g. some Termux/Kaggle kernels — the mode is recorded
  on every `EvalResult`). `/proc` is not mounted in the sandbox (some stdlib
  corners unavailable). A candidate that deliberately reads its own file to
  steal the per-run token and forge a pass is outside the scoring threat
  model (blatant in logs; contained by the sandbox regardless). Kernel-level
  escapes are out of scope. **Policy unchanged: untrusted candidates at scale
  run in a throwaway container/VM.**

## Gate D — Baselines and metrics (PASS)

Identical rules, one command each (`benchmarks/run_heldout.py`), deterministic
fixed-string baselines, results saved as JSON:

| baseline | split | pass@1 | funnel |
|---|---|---|---|
| empty (`pass` body) | all 24 | 0/24 = 0.000 | 24 assertion |
| none (`return None`) | all 24 | 0/24 = 0.000 | 24 assertion |
| oracle (reference) | all 24 | 24/24 = 1.000 | 24 pass |

Saved: `experiments/heldout_empty.json`, `experiments/heldout_none.json`
(oracle run verified live; JSON in /tmp during the session).
No toy-memorization numbers are reported as coding competence — there are no
model scores in M4 at all.

## Gate E — Regression and evidence (PASS)

New `tests/test_heldout.py`: 21 tests covering benchmark integrity (version,
24 tasks, 12/12 splits, hash re-verification, frozen flag), scoring paths
(oracle 1.0, both baselines 0.0, wrong/exception/syntax/timeout/spoof),
contamination controls (0/24 vs fingerprints, verbatim positive control,
0/24 vs M2 corpus), and safety boundaries (sandbox mode recorded, host-read
/ oldroot / network blocked in pivot mode, cleared env, fork-bomb contained).

Full suite, this session, all green:
- tests/test_pipeline.py: 26 OK (unchanged)
- tests/test_decontam.py: 6 OK (unchanged)
- tests/test_train.py: 8 OK (unchanged)
- tests/test_heldout.py: 21 OK (new)
- benchmarks/eval_harness.py self-test: PASS (oracle 1.0, empty 0.0)
- benchmarks/test_harness_adversarial.py: OK (see Gate C)

## What changed in the repo (vs 60887ad)

- `benchmarks/eval_harness.py`: per-run completion token; pivot_root userns
  sandbox with probe + `plain` fallback; `EvalResult.sandbox` field; Popen +
  process-group kill on timeout; RLIMIT_NPROC + RLIMIT_FSIZE; sbin in
  candidate PATH (for `pivot_root`/`mount`); honest docstrings.
- `benchmarks/build_heldout.py` (new): 24 original tasks + quarantined
  oracles + frozen JSON builder.
- `benchmarks/forge_heldout_v1.json` (new): the frozen benchmark.
- `benchmarks/run_heldout.py` (new): reproducible baseline runner.
- `benchmarks/quarantine/README.md`: FORGE-Heldout quarantine + audit record.
- `tests/test_heldout.py` (new): 21 regression tests.
- `experiments/heldout_empty.json`, `experiments/heldout_none.json`: baseline
  evidence. This report: `experiments/EXP-004-m4-report.md`.

## Conditions carried forward (unchanged)

1. M5 pretraining stays BLOCKED: training-corpus source decision still open
   (the-stack-v2 gated terms need Mohnish's own acceptance; codeparrot-clean
   fallback verified) AND a Kaggle GPU smoke run must validate Gate E estimates.
2. Previous reports untouched; no scores fabricated; no gate weakened.
3. Throwaway-VM policy for untrusted candidates at scale remains, even with
   the pivot sandbox.
