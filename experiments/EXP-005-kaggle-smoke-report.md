# EXP-005 — Kaggle T4 ×2 Smoke Run & Account Workflow

Date: 2026-10-24. Repo: ~/workspace/forge.
Code commit: `6775404` (feat: DDP multi-GPU path). Notebook commit: `5dcbc3f`.

## Status: PREPARED — Kaggle execution NOT RUN

No GPU exists in this environment, so **no Kaggle execution is claimed**.
Everything below the "Kaggle run" line is preparation + local verification.
The GPU-results section is explicitly marked NOT RUN and must be filled in
from real logs after Mohnish runs the notebook.

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

## 5. GPU results — NOT RUN (fill in after Mohnish's run)

| item | value |
|---|---|
| GPU model / count | _pending_ |
| CUDA usable | _pending_ |
| Run A tok/s | _pending_ |
| Run B tok/s, peak mem | _pending_ |
| Run C (DDP) scaling | _pending_ |
| MFU (FP32) | _pending_ |
| 125M projection | _pending_ |

## 6. Failures, limitations, next decision

- **Environment failure (local):** TCP-based torch.distributed rendezvous
  does not work in this VM (loopback handshake corrupted) — DDP was
  validated via `file://` rendezvous instead. Kaggle NCCL is unaffected
  by this; Run C will tell us the truth.
- **Not validated:** NCCL 2-GPU scaling; any GPU number in this report.
- **Notebook caveat:** `git clean -fdx` in the fetch cell wipes untracked
  files in the clone (regenerates toy data after) — by design, but noted.
- **Next decision:** after Mohnish pastes `gpu_smoke_results.json`, revise
  the EXP-003 Gate E table with measured MFU. If Run B MFU ∈ 25–35%, the
  M5 plan (125M v0 ≈ 23–32 T4-hours) is validated; if far off, revise
  before spending quota.

## 7. Gates (unchanged)

- **M5 pretraining: BLOCKED.** Corpus decision is separate and still open
  (the-stack-v2 gated terms need Mohnish's own HF acceptance).
- No pretraining started; no GPU execution claimed; no scores fabricated.
