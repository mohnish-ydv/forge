# EXP-005 — Kaggle T4 ×2 Smoke Run & Account Workflow

Date: 2026-10-24. Repo: ~/workspace/forge.
Code commit: `6775404` (feat: DDP multi-GPU path). Notebook commit: `5dcbc3f`.
Kaggle run: 2026-10-24, notebook pinned to GitHub snapshot `fa63122` (param_hash CUDA fix).

## Status: RUN — real Kaggle GPU results below (section 5)

Run A/B/C all completed on Kaggle. Total GPU time ≈ 4 min (9.5 + 173.9 + 41.8 s).
Raw results archived at `experiments/gpu_smoke_results.json` (as uploaded by
Mohnish); independently verified by `experiments/analyze_smoke_results.py`
(26 ok, 0 failures after two analyzer bugs were fixed — see section 6).
**Gate E verdict: MFU assumption VALIDATED (30.9% ∈ 25–35%), but the
EXP-003 hour estimate is NOT validated** — it assumed the 65 TFLOPS FP16
peak, while training is FP32 (8.1 TFLOPS peak), so the honest FP32
projection is ~218 GPU-hours for 125M/2.5B tokens, not 23–32 h. Mixed
precision (FP16 AMP) is now a prerequisite for M5, not an optimization.

## 1. What was verified locally (this VM, CPU-only)

- Commit `7aabde8` verified; working tree was clean before changes.
- `src/train.py --device` (from 7aabde8) re-verified bit-identical:
  40-step smoke reproduces M3 exactly (`final_train_loss 17.793315887451172`,
  `param_hash 1200.7129848422692`). Checkpoint v1 format unchanged
  (`map_location="cpu"` on load).
- New `--ddp` flag (commit `6775404`): single-process path unchanged
  (bit-identical re-verified after the change). DDP logic validated with
  2 processes on CPU/gloo via `FORGE_DIST_INIT_METHOD=file:///tmp/forge_ddp_init`:
  loss 91.3 → 73.1 over 12 steps, no hangs, rank-0-only checkpoint
  (`ckpt_000012.pt`, v1) loads into a plain model, no duplicate SUMMARY.
- 8/8 `tests/test_train.py` pass after all changes.
- Notebook: valid nbformat 4.4 JSON, 15 cells, all code cells parse,
  execution order reviewed (env → fetch+pin → toy data → Run A → Run B →
  optional Run C → results JSON). No secrets, no stored outputs.

## 2. Two-GPU handling (honest)

- Current training code is single-process and uses **cuda:0 only**
  (`torch.device("cuda")`). Seeing 2 GPUs in `nvidia-smi` does not mean
  both are used — the notebook's env cell reports `device_count()` and
  the results JSON records it.
- `--ddp` (torchrun + DDP, NCCL on CUDA) is the multi-GPU path. Its *logic*
  is validated on CPU/gloo; **NCCL on 2×T4 is NOT validated** — the
  notebook's optional Run C is the validation vehicle. Run C compares
  aggregate tok/s against 2× the single-GPU rate and reports scaling
  efficiency (1 − comm overhead fraction).
- The M3 baseline is untouched: `--ddp` off ⇒ byte-identical behavior.

## 3. Notebook design (commit 5dcbc3f)

- `FORGE_COMMIT='6775404'` pinned; fetch cell re-enforces it on reruns.
- Env cell: python/torch/CUDA/cuDNN, GPU count, per-GPU name + VRAM.
- Runs A/B use explicit `--device cuda` and assert `device=cuda` appears
  in the log — a CUDA run cannot silently fall back to CPU (train.py also
  raises if CUDA is unavailable).
- Results → `/kaggle/working/gpu_smoke_results.json`: hardware, pinned +
  actual commit SHA, per-run config/elapsed/tok/s/peak-mem/losses/errors.
- MFU methodology: **measured** = tok/s, elapsed, peak mem, losses.
  **Estimated** = FLOPs/token ≈ 6N + 12·L·d·S (N exact from config;
  ±10% methodology band). **Peak** = FP32 (T4 8.1 TFLOPS, P100 9.3) —
  training is FP32 with no AMP, so the 65 TFLOPS FP16-tensor figure does
  NOT apply. MFU = tok/s × FLOPs/token ÷ FP32 peak. Short-run noise noted.

## 4. Kaggle account workflow (for Mohnish)

Phone-friendly path (simplest first):

1. **Push the repo to GitHub** (private is fine): on your device,
   `git remote add origin <url>` then `git push -u origin master`
   from `~/workspace/forge` (commits up to `5dcbc3f`).
2. **Kaggle → Code → New Notebook → ⋮ menu → Upload** the file
   `notebooks/forge_m5_gpu_smoke.ipynb` from the repo.
3. **Settings (right panel): Accelerator → GPU T4 x2**; turn **Internet ON**
   (the fetch cell needs it for `git clone`).
4. Set `REPO_URL` in cell 3 to your GitHub repo URL. **Run all.**
   Budget: < 30 min GPU.
5. **Download** `/kaggle/working/gpu_smoke_results.json` (right panel →
   `/kaggle/working`) and paste its contents back here.

Prerequisite: Kaggle requires phone verification before GPU access is
unlocked — if the accelerator dropdown is greyed out, that's why.

CLI/API alternative (later, optional):
- `pip install kaggle`; create the API token at kaggle.com → Settings →
  API → "Create New Token" → saves `kaggle.json`. **Keep it private on
  your own device** — never paste it into chat or commit it.
- `mkdir kpush && cp notebooks/forge_m5_gpu_smoke.ipynb kpush/`
- `kpush/kernel-metadata.json`:
  ```json
  {
    "id": "<your-username>/forge-m5-gpu-smoke",
    "title": "FORGE M5 GPU smoke",
    "code_file": "forge_m5_gpu_smoke.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true"
  }
  ```
- `kaggle kernels push -p kpush` (pushes **and runs** the notebook),
  `kaggle kernels status <username>/forge-m5-gpu-smoke`,
  `kaggle kernels output <username>/forge-m5-gpu-smoke -p out/`
  (fetches logs + `/kaggle/working` files).
- **T4 ×2 identifier:** there is no documented metadata field for 2×T4 —
  `machine_shape` (e.g. `"NvidiaTeslaT4"`) selects a *single*-GPU type in
  some setups, but **2×T4 is selected in the notebook Settings UI only**.
  After a CLI push, open the notebook in the browser and set
  Accelerator → GPU T4 x2 before running.

## 5. GPU results — RUN 2026-10-24 (Kaggle, real hardware)

| item | value |
|---|---|
| GPU model / count | 2× Tesla T4, 15,636 MB VRAM each |
| Software | torch 2.11.0+cu128, CUDA 12.8, Python 3.13.15 |
| Commit pin | pinned = actual HEAD = `fa63122` (param_hash CUDA fix) |
| CUDA usable | yes (asserted in env cell; all runs on cuda:0 / DDP) |
| Run A tok/s | 2,146 (tiny 1.02M, 40 steps; loss 17.85/17.94 — reproduces M3 CPU 17.79/18.24) |
| Run B tok/s, peak mem | 14,133 tok/s, 3,488 MB (25.31M, 300 steps, batch 16×512) |
| Run C (DDP) scaling | aggregate 15,664 tok/s vs single-GPU 14,133 → ~11% faster aggregate, ~55% scaling efficiency — CONFOUNDED (40 vs 300 steps, 2× global batch); not a clean scaling measurement |
| MFU (FP32) | Run B 30.9% (independently recomputed, agrees with notebook) |
| MFU (FP32) Run C | 17.1% per GPU (notebook's 0.342 divides 2-GPU throughput by a single GPU's peak — overstated 2×, corrected by the analyzer) |
| 125M projection | ~218 GPU-hours FP32 (125M/2.5B tokens) — OUTSIDE the EXP-003 band 23–32 h |

Run B loss 0.173/0.176 is memorization of the repetitive toy fixture by a
25M model over 300 steps, not divergence (train/val ratio 1.02). Run A tok/s
is a tiny-model artifact (512 tokens/step, val eval dominating), not a GPU
verdict — Run B is the measurement that matters.

## 6. Failures, limitations, next decision

- **Environment failure (local):** TCP-based torch.distributed rendezvous
  does not work in this VM (loopback handshake corrupted) — DDP was
  validated via `file://` rendezvous instead. Kaggle NCCL is unaffected
  by this; Run C confirmed DDP works on Kaggle (both GPUs used).
- **Analyzer bugs fixed during review (2026-10-24):** the review tool's
  absolute loss band (5–45) falsely failed Run B/C (low loss on this
  fixture is memorization, not divergence) — replaced with
  finite/positive + train/val-agreement checks; its hardcoded pin was
  stale (d6089ae0 → fa63122); its Run C MFU divided 2-GPU throughput by
  a single GPU's peak (0.342 → honest 0.171 per GPU). After fixes:
  26 ok, 0 failures.
- **Notebook caveat:** `git clean -fdx` in the fetch cell wipes untracked
  files in the clone (regenerates toy data after) — by design, but noted.
- **Notebook caveat 2:** the notebook's Run C MFU (0.342) is overstated 2×
  (same denominator error as the analyzer had); use 17.1% per GPU.
- **Next decision (Gate E verdict):** the MFU assumption (25–35%) is
  validated at 30.9%, but the EXP-003 hour estimate assumed FP16 peak
  (65 TFLOPS) while training is FP32 (8.1 TFLOPS). At FP32, 125M/2.5B
  costs ~218 GPU-hours — ~7 weeks of Kaggle's ~30 GPU-h/week quota for
  one run. **Do not start M5 pretraining at FP32.** Next step: implement
  FP16 mixed precision (autocast + GradScaler) in `src/train.py`, then a
  short GPU smoke re-run of Run B with AMP to measure FP16 MFU. At
  similar MFU on 65 TFLOPS peak, the projection returns to ~27 h,
  inside the original band. Corpus decision (the-stack-v2 terms vs
  codeparrot-clean fallback) remains open in parallel.

## 7. Gates (unchanged)

- **M5 pretraining: BLOCKED.** (a) Corpus decision still open
  (the-stack-v2 gated terms need Mohnish's own HF acceptance).
  (b) NEW: FP32 compute cost too high (~218 h); AMP + FP16 smoke
  re-validation required first.
- No pretraining started; no scores fabricated. GPU numbers above are
  measured, projections are labeled estimates.
