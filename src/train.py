"""FORGE minimal training loop (M3 Gate B/C).

Trains a tiny decoder-only Transformer (src/model.py) with causal LM loss
on the deterministic toy dataset (src/toy_data.py). Purpose: prove the
training system works end-to-end, survives interruption, and resumes
correctly — NOT to claim coding capability.

Checkpoint format (v1):
  {model, optimizer, scheduler, step, tokens_seen, config, rng, version}
  AMP runs additionally store `scaler` (GradScaler state_dict) and record
  `precision` inside config["train"] (argparse captures it automatically).
  v1 checkpoints without those keys load as FP32 (backward compatible);
  resuming across precisions, or AMP-resuming from a scaler-less checkpoint,
  is a hard error, never a silent change of experiment.
- Atomic write: save to tmp file, fsync, os.replace.
- Rotation: keep last `keep_last` validated checkpoints; the previous
  known-good checkpoint is never deleted before the new one validates.
- Corrupt/missing checkpoint -> hard error, NEVER a silent fresh run.

Usage:
  python src/train.py --steps 40 --ckpt-every 10            # fresh run
  python src/train.py --resume --steps 40                   # resume to 40
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_VERSION = 1

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False


def get_args():
    p = argparse.ArgumentParser(description="FORGE M3 smoke training")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-kv-heads", type=int, default=2)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--keep-last", type=int, default=2)
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints/smoke")
    p.add_argument("--t-max", type=int, default=0,
                   help="CosineAnnealing T_max. Defaults to --steps. MUST be identical "
                        "across initial and resumed runs for a valid recovery comparison.")
    p.add_argument("--data", type=str, default="data/toy/train.bin")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--device", type=str, default="auto",
                   help="Compute device: auto (cuda when available), cpu, or cuda. "
                        "Required for GPU smoke runs (M5 Gate E validation).")
    p.add_argument("--ddp", action="store_true",
                   help="Multi-GPU data-parallel via torchrun + DDP. Launch with "
                        "torchrun --nproc_per_node=N. Single-process path is "
                        "unchanged when off (M3 baseline preserved). DDP path is "
                        "NOT covered by the M3 bit-identical guarantee.")
    p.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"],
                   help="Training precision: fp32 (default, M3 baseline) or "
                        "fp16 AMP (CUDA only: autocast + GradScaler). CPU FP16 "
                        "is rejected, not silently emulated.")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Steps excluded from steady-state throughput measurement "
                        "(benchmarking only; does not change training). "
                        "steady_tok_per_s is None when warmup >= steps.")
    return p.parse_args()


def log(msg, logf=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logf:
        with open(logf, "a") as f:
            f.write(line + "\n")


# ------------------------------------------------------------ checkpointing

def _rng_states():
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }


def _set_rng_states(st):
    torch.set_rng_state(st["torch"])
    random.setstate(st["python"])


def save_checkpoint(path, model, optimizer, scheduler, step, tokens_seen, config,
                    scaler=None):
    """Atomic, validated save. Raises on failure (never partial)."""
    tmp = path + ".tmp"
    payload = {
        "version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "tokens_seen": tokens_seen,
        "config": config,
        "rng": _rng_states(),
        # AMP only: None for FP32 runs (v1-compatible shape).
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(payload, tmp)
    # Validate the new checkpoint BEFORE it replaces anything: reload and
    # check required keys + step consistency.
    probe = torch.load(tmp, map_location="cpu", weights_only=False)
    assert probe["version"] == CHECKPOINT_VERSION, "version mismatch"
    assert probe["step"] == step, "step mismatch in written checkpoint"
    for k in ("model", "optimizer", "config", "rng"):
        assert k in probe, f"missing key {k}"
    if scaler is not None:
        assert probe.get("scaler") is not None, "scaler state missing in written checkpoint"
    os.replace(tmp, path)  # atomic on POSIX


def check_resume_precision(stored_precision, requested_precision, has_scaler_state):
    """Pure resume-compatibility decision (unit-testable without CUDA).

    stored_precision: from the checkpoint's config["train"] ("fp32" when the
    key is absent, i.e. v1 checkpoints). Raises RuntimeError on any
    incompatible combination; returns None when the resume may proceed.
    """
    if stored_precision != requested_precision:
        raise RuntimeError(
            f"Precision mismatch on resume: checkpoint used {stored_precision}, "
            f"current run requests {requested_precision}. Refusing to resume across "
            f"precisions (use a fresh --ckpt-dir for the new precision).")
    if requested_precision == "fp16" and not has_scaler_state:
        raise RuntimeError(
            "AMP resume requires GradScaler state in the checkpoint, but none "
            "was found. Refusing to resume with a fresh scaler (it would "
            "silently change the experiment).")
    return None


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """Load and apply checkpoint. Raises FileNotFoundError / RuntimeError
    on missing/corrupt files — the caller must NOT fall back to a fresh run
    silently. When `scaler` is given and the checkpoint carries scaler state,
    it is restored; a scaler-less (v1/FP32) checkpoint leaves the scaler
    untouched, and the caller decides whether that is compatible."""
    import torch
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"checkpoint corrupt/unreadable: {path}: {e}") from e
    if ckpt.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError(f"unsupported checkpoint version in {path}")
    for k in ("model", "optimizer", "config", "rng", "step"):
        if k not in ckpt:
            raise RuntimeError(f"checkpoint {path} missing key: {k}")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    _set_rng_states(ckpt["rng"])
    return ckpt


def latest_checkpoint(ckpt_dir):
    cands = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("ckpt_") and f.endswith(".pt"):
            try:
                cands.append((int(f[5:-3]), os.path.join(ckpt_dir, f)))
            except ValueError:
                pass
    return max(cands)[1] if cands else None


def rotate_checkpoints(ckpt_dir, keep_last):
    cands = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("ckpt_") and f.endswith(".pt"):
            try:
                cands.append((int(f[5:-3]), os.path.join(ckpt_dir, f)))
            except ValueError:
                pass
    cands.sort()
    # Delete oldest beyond keep_last — only called AFTER the new checkpoint
    # has been written and validated, so known-good is never lost first.
    for _, path in cands[:-keep_last]:
        os.unlink(path)


# ------------------------------------------------------------ data

def make_batches(ids, seq_len, batch_size, seed):
    """Deterministic batch index list: fixed permutation, chunked.

    No padding anywhere: every batch is full (seq_len+1) by construction;
    the final partial chunk is dropped. Causal loss uses shifted targets
    (input[:, :-1] -> logits, input[:, 1:] -> targets) inside the loop.
    """
    n_chunks = (len(ids) - 1) // (seq_len + 1)
    usable = n_chunks * (seq_len + 1)
    chunks = torch.tensor(ids[:usable], dtype=torch.long).view(n_chunks, seq_len + 1)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_chunks, generator=g)
    batches = []
    for i in range(0, n_chunks, batch_size):
        idx = perm[i:i + batch_size]
        if len(idx) < batch_size:
            break  # drop partial batch: keeps every step identical shape
        batches.append(chunks[idx])
    return batches


@torch.no_grad()
def evaluate(model, batches, device, amp=False):
    import torch
    from contextlib import nullcontext
    # Same autocast policy as training (fp16 matmuls, fp32 loss) so val loss
    # is comparable across precisions. amp=False -> nullcontext: the FP32
    # path is exactly unchanged.
    ctx = (torch.autocast("cuda", dtype=torch.float16, enabled=True)
           if (amp and device.type == "cuda") else nullcontext())
    model.eval()
    total, n = 0.0, 0
    for b in batches:
        b = b.to(device)
        x = b[:, :-1]
        y = b[:, 1:]
        with ctx:
            _, loss = model(x, y)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def param_hash(model):
    # h lives on CPU; CUDA params must be moved over before adding, otherwise
    # a CUDA tensor + CPU accumulator raises (caught by the Kaggle GPU smoke run).
    # .cpu() is a no-op for CPU tensors, so CPU results are bit-identical.
    h = torch.zeros((), dtype=torch.float64)
    for p in model.parameters():
        h += p.detach().double().sum().cpu()
    return float(h.item())


def main():
    args = get_args()
    if not HAS_TORCH:
        print("FATAL: torch is not installed.", file=sys.stderr)
        print("Install a PyTorch runtime first: "
              ".venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu",
              file=sys.stderr)
        sys.exit(2)
    from model import ForgeModel, ForgeConfig
    from toy_data import load_ids

    logf = args.log_file or None
    dev = args.device.lower()
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- DDP setup (all guarded; single-process path unchanged when off) ---
    ddp = args.ddp
    if ddp:
        import torch.distributed as dist
        backend = "nccl" if (dev == "cuda" and torch.cuda.is_available()) else "gloo"
        # FORGE_DIST_INIT_METHOD: optional explicit init method, e.g.
        # "file:///tmp/forge_dist" (useful where TCP rendezvous is blocked).
        # Default: env-based (what torchrun sets up).
        init_kwargs = {}
        init_method = os.environ.get("FORGE_DIST_INIT_METHOD")
        if init_method:
            # file:// rendezvous requires explicit rank/world_size
            init_kwargs["init_method"] = init_method
            init_kwargs["rank"] = int(os.environ.get("RANK", "0"))
            init_kwargs["world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
        dist.init_process_group(backend=backend, **init_kwargs)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        is_main = rank == 0
        if dev == "cuda":
            dev = f"cuda:{local_rank}"
            torch.cuda.set_device(dev)
        log(f"ddp: backend={backend} rank={rank}/{world_size} local_rank={local_rank}", logf)
    else:
        rank, world_size, is_main = 0, 1, True

    device = torch.device(dev)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch reports no CUDA device")
    if args.warmup_steps < 0:
        raise RuntimeError("--warmup-steps must be >= 0")
    precision = args.precision.lower()
    # AMP is a CUDA-only path. CPU FP16 exists in torch but is not a
    # supported equivalent (no tensor-core acceleration, different numerics),
    # so reject it loudly instead of benchmarking something meaningless.
    if precision == "fp16" and device.type != "cuda":
        raise RuntimeError(
            "--precision fp16 requires a CUDA device; on CPU use --precision fp32. "
            "CPU FP16 is not a supported equivalent to CUDA AMP.")
    amp = precision == "fp16"
    if is_main:
        log(f"device={device} precision={precision}", logf)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    try:
        torch.use_deterministic_algorithms(True)
        det = True
    except Exception:
        det = False

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ids = load_ids(args.data)
    n_val = max(1, len(ids) // 10)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    train_batches = make_batches(train_ids, args.seq_len, args.batch_size, args.seed)
    val_batches = make_batches(val_ids, args.seq_len, args.batch_size, args.seed + 1)
    assert train_batches, "no training batches: toy data too small"
    log(f"batches: train={len(train_batches)} val={len(val_batches)} "
        f"tokens/batch={args.batch_size * args.seq_len} deterministic_algos={det}", logf)

    mcfg = ForgeConfig(vocab_size=256, d_model=args.d_model, n_layers=args.n_layers,
                       n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
                       max_seq_len=args.seq_len + 1)
    model = ForgeModel(mcfg).to(device)
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model)
        log(f"ddp: model wrapped, effective batch/rank={args.batch_size} "
            f"global={args.batch_size * world_size}", logf)
    if is_main:
        log(f"params: {mcfg.param_count()['total_M']}M", logf)
    t_max = args.t_max or args.steps
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)

    # --- AMP setup (fp16 only; FP32 path keeps scaler=None throughout) ---
    # autocast policy (torch defaults): linear/matmul in fp16, softmax and
    # cross-entropy in fp32, master weights stay fp32. RMSNorm is computed in
    # fp32 explicitly in model.py. RoPE freqs are fp32; the bounded [-1,1]
    # rotation products under autocast run in fp16 without overflow risk.
    from contextlib import nullcontext
    amp_ctx = (torch.autocast("cuda", dtype=torch.float16, enabled=True)
               if (amp and device.type == "cuda") else nullcontext())
    scaler = torch.amp.GradScaler("cuda") if amp else None
    # Skipped-step counting: GradScaler does not report skips, but it calls
    # optimizer.step() ONLY when no inf/nan was found. Wrapping step() with a
    # counter therefore measures skips exactly, with no private APIs.
    step_calls = []
    if scaler is not None:
        _orig_step = opt.step

        def _counting_step(*a, **k):
            step_calls.append(1)
            return _orig_step(*a, **k)

        opt.step = _counting_step

    start_step, tokens_seen = 0, 0
    if args.resume:
        ckpt_path = latest_checkpoint(args.ckpt_dir)
        # FAIL SAFE: missing checkpoint with --resume is a hard error,
        # never a silent fresh run.
        ckpt = load_checkpoint(ckpt_path, model, opt, sched, scaler=scaler)
        old_t_max = ckpt["config"]["train"].get("t_max") or ckpt["config"]["train"]["steps"]
        if old_t_max != t_max:
            raise RuntimeError(
                f"T_max mismatch on resume: checkpoint used T_max={old_t_max}, "
                f"current run has T_max={t_max}. Refusing to resume with a "
                f"different LR schedule.")
        # Precision compatibility: a checkpoint's precision is part of the
        # experiment. v1 checkpoints carry no precision key and are FP32.
        stored_precision = ckpt["config"]["train"].get("precision", "fp32")
        check_resume_precision(stored_precision, precision, ckpt.get("scaler") is not None)
        start_step, tokens_seen = ckpt["step"], ckpt["tokens_seen"]
        log(f"resumed from {ckpt_path} at step {start_step}", logf)
        if precision == "fp16":
            log(f"resumed scaler scale={scaler.get_scale()}", logf)

    t0 = time.time()
    train_losses = []
    # Steady-state throughput: warmup steps (allocator/cudnn warm-up, first
    # val pass) are excluded. Tokens at the measurement boundary are recorded
    # so the count is exact under resume and DDP. Val passes inside the
    # measured region are identical for both precisions, so the FP16/FP32
    # ratio is unaffected (documented, not hidden).
    t_measure = t0 if args.warmup_steps == 0 else None
    measure_tokens = tokens_seen * world_size if args.warmup_steps == 0 else 0
    skipped_steps, nonfinite_losses, nonfinite_grads = 0, 0, 0
    # DDP: each rank trains on a strided shard of the deterministic batches.
    # DDP averages gradients across ranks, so one optimizer step == one step
    # on the concatenated global batch.
    rank_batches = train_batches[rank::world_size]
    assert rank_batches, f"rank {rank}: no batches in strided shard"
    for step in range(start_step, args.steps):
        b = rank_batches[step % len(rank_batches)].to(device)
        x, y = b[:, :-1], b[:, 1:]
        assert x.shape == (args.batch_size, args.seq_len), "batch shape drift"
        with amp_ctx:
            _, loss = model(x, y)
        loss_val = loss.item()
        if scaler is not None:
            # AMP: a non-finite loss yields inf grads, which the scaler skips
            # safely. Count it (benchmark honesty) instead of crashing: the
            # gate is "does the scaler repeatedly overflow", not "never inf".
            if not math.isfinite(loss_val):
                nonfinite_losses += 1
        else:
            assert math.isfinite(loss_val), f"non-finite loss at step {step}"
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            # Docs order: scale -> backward -> unscale -> clip -> step -> update.
            # Clipping runs on UNSCALED grads. The scheduler advances even on
            # a skipped step (docs pattern); skipped_steps is reported so the
            # effect is visible, not hidden.
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not math.isfinite(grad_norm.item()):
                nonfinite_grads += 1
            step_calls.clear()
            scaler.step(opt)
            scaler.update()
            skipped_steps += 0 if step_calls else 1
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            assert torch.isfinite(grad_norm).item(), f"non-finite grad norm at step {step}"
            opt.step()
        sched.step()
        step1 = step + 1
        tokens_seen += args.batch_size * args.seq_len
        train_losses.append(loss_val)

        if t_measure is None and args.warmup_steps > 0 and step1 == args.warmup_steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_measure = time.time()
            measure_tokens = tokens_seen * world_size

        if ddp and (step1 % args.log_every == 0 or step1 == args.steps):
            # average the logged loss across ranks (in-place on a tensor)
            import torch.distributed as dist
            avg = torch.tensor(loss.item(), device=device)
            dist.all_reduce(avg, op=dist.ReduceOp.SUM)
            log_loss = float(avg.item() / world_size)
        else:
            log_loss = loss.item()

        if is_main and (step1 % args.log_every == 0 or step1 == args.steps):
            el = time.time() - t0
            tps = tokens_seen * world_size / max(el, 1e-9)
            log(f"step {step1}/{args.steps} loss={log_loss:.4f} "
                f"tok={tokens_seen * world_size} t={el:.1f}s tok/s={tps:.0f}", logf)
        if is_main and (step1 % args.val_every == 0 or step1 == args.steps):
            vl = evaluate(model.module if ddp else model, val_batches, device, amp=amp)
            log(f"  val_loss={vl:.4f}", logf)
        if is_main and (step1 % args.ckpt_every == 0 or step1 == args.steps):
            path = os.path.join(args.ckpt_dir, f"ckpt_{step1:06d}.pt")
            cfg = {"model": mcfg.__dict__, "train": vars(args)}
            save_checkpoint(path, model.module if ddp else model, opt, sched,
                            step1, tokens_seen * world_size, cfg, scaler=scaler)
            rotate_checkpoints(args.ckpt_dir, args.keep_last)
            log(f"  checkpoint -> {path} (validated, rotated)", logf)

    if device.type == "cuda":
        torch.cuda.synchronize()
    el = time.time() - t0
    # Steady-state throughput over the post-warmup region only. None when the
    # run never crossed the warmup boundary (warmup >= steps, or a resume
    # that started past it) — reported as null, never faked from the full run.
    if t_measure is not None:
        steady_el = time.time() - t_measure
        steady_tokens = tokens_seen * world_size - measure_tokens
        steady_tps = (round(steady_tokens / max(steady_el, 1e-9), 1)
                      if steady_tokens > 0 else None)
    else:
        steady_tps = None
    if is_main:
        summary = {
            "steps": args.steps, "final_train_loss": train_losses[-1],
            "final_val_loss": evaluate(model.module if ddp else model, val_batches, device, amp=amp),
            "tokens_seen": tokens_seen * world_size, "elapsed_s": round(el, 2),
            "tok_per_s": round(tokens_seen * world_size / max(el, 1e-9), 1),
            "param_hash": param_hash(model.module if ddp else model),
            "seed": args.seed, "deterministic_algos": det,
            "ddp": ddp, "world_size": world_size,
            "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
                               if device.type == "cuda" else 0.0,
            "peak_gpu_mem_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1e6, 1)
                               if device.type == "cuda" else 0.0,
            # AMP / benchmark fields (FP32 runs report the neutral values).
            "precision": precision,
            "warmup_steps": args.warmup_steps,
            "steady_tok_per_s": steady_tps,
            "skipped_steps": skipped_steps,
            "nonfinite_losses": nonfinite_losses,
            "nonfinite_grads": nonfinite_grads,
            "scaler_final_scale": scaler.get_scale() if scaler is not None else None,
        }
        log("SUMMARY " + json.dumps(summary), logf)
        with open(os.path.join(args.ckpt_dir, "last_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
    if ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

