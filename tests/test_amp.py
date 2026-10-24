"""EXP-006: AMP (FP16) training-path tests. Run with .venv python.

Covers what is testable without CUDA:
 1. --precision validation (choices + CPU-FP16 rejection)
 2. FP32 summary carries precision/skipped/scaler labels; FP32 evidence
    stays in its own directory, labelled fp32
 3. GradScaler state_dict round-trip (serialization, no CUDA needed)
 4. Checkpoint scaler plumbing: FP32 (v1-style, scaler=None) loads cleanly;
    scaler state survives save/load
 5. Resume-precision compatibility decisions (pure function)
 6. CPU FP32 smoke reproduces the exact M3 canonical values
    (train 17.793315887451172, val 18.236953735351562,
     param_hash 1200.7129848422692) -- the FP32 baseline is bit-identical
 7. DDP on CPU/gloo: only rank 0 writes SUMMARY/checkpoints
    (exactly one SUMMARY line in merged torchrun output)

Explicitly NOT testable here (no GPU in this environment -- marked, not
claimed): actual CUDA AMP training, autocast numerics, skipped-step
behavior under real inf grads, FP16 tok/s, and AMP checkpoint resume on
CUDA. Those are covered by the Kaggle benchmark notebook (EXP-006) and its
fresh-process resume cell.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from model import ForgeModel, ForgeConfig
from train import (save_checkpoint, load_checkpoint, check_resume_precision)

ROOT = os.path.join(os.path.dirname(__file__), "..")
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
TRAIN_PY = os.path.join(ROOT, "src", "train.py")
DATA_BIN = os.path.join(ROOT, "data", "toy", "train.bin")


def run_train(args, cwd=ROOT, env=None, timeout=300):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run([VENV_PY, TRAIN_PY] + args, capture_output=True,
                          text=True, timeout=timeout, cwd=cwd, env=e)


def tiny_model():
    cfg = ForgeConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                      n_kv_heads=2, max_seq_len=33)
    torch.manual_seed(7)
    return ForgeModel(cfg), cfg


class TestPrecisionValidation(unittest.TestCase):
    def test_invalid_precision_rejected(self):
        r = run_train(["--precision", "bf16", "--steps", "1"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("invalid choice", r.stderr)

    def test_fp16_on_cpu_rejected_clearly(self):
        r = run_train(["--precision", "fp16", "--device", "cpu", "--steps", "1"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("requires a CUDA device", r.stderr)

    def test_fp16_auto_on_cpu_only_rejected(self):
        # --device auto resolves to cpu here, so fp16 must still be rejected.
        r = run_train(["--precision", "fp16", "--device", "auto", "--steps", "1"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("requires a CUDA device", r.stderr)


class TestScalerCheckpointPlumbing(unittest.TestCase):
    def test_scaler_state_dict_roundtrip(self):
        s = torch.amp.GradScaler("cuda")
        s2 = torch.amp.GradScaler("cuda")
        s2.load_state_dict(s.state_dict())
        self.assertEqual(s2.state_dict(), s.state_dict())

    def test_fp32_checkpoint_loads_with_scaler_arg(self):
        # v1-style: saved with scaler=None. Loading with a scaler passed must
        # leave the scaler untouched (no state to restore) and succeed.
        m, cfg = tiny_model()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ckpt_000004.pt")
            save_checkpoint(path, m, opt, sched, 4, 2048,
                            {"model": cfg.__dict__, "train": {"precision": "fp32"}},
                            scaler=None)
            scaler = torch.amp.GradScaler("cuda")
            before = scaler.state_dict()
            m2, _ = tiny_model()
            opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=4)
            ckpt = load_checkpoint(path, m2, opt2, sched2, scaler=scaler)
            self.assertEqual(scaler.state_dict(), before)
            self.assertIsNone(ckpt.get("scaler"))

    def test_scaler_state_survives_save_load(self):
        m, cfg = tiny_model()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4)
        scaler = torch.amp.GradScaler("cuda")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ckpt_000004.pt")
            save_checkpoint(path, m, opt, sched, 4, 2048,
                            {"model": cfg.__dict__, "train": {"precision": "fp16"}},
                            scaler=scaler)
            fresh = torch.amp.GradScaler("cuda")
            m2, _ = tiny_model()
            opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=4)
            ckpt = load_checkpoint(path, m2, opt2, sched2, scaler=fresh)
            self.assertIsNotNone(ckpt.get("scaler"))
            self.assertEqual(fresh.state_dict(), scaler.state_dict())


class TestResumePrecision(unittest.TestCase):
    def test_match_fp32_ok(self):
        self.assertIsNone(check_resume_precision("fp32", "fp32", False))

    def test_match_fp16_with_scaler_ok(self):
        self.assertIsNone(check_resume_precision("fp16", "fp16", True))

    def test_mismatch_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Precision mismatch"):
            check_resume_precision("fp32", "fp16", False)
        with self.assertRaisesRegex(RuntimeError, "Precision mismatch"):
            check_resume_precision("fp16", "fp32", True)

    def test_fp16_without_scaler_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "GradScaler state"):
            check_resume_precision("fp16", "fp16", False)

    def test_v1_checkpoint_defaults_to_fp32(self):
        # v1 checkpoints carry no precision key; the caller defaults to fp32.
        stored = {}.get("precision", "fp32")
        self.assertIsNone(check_resume_precision(stored, "fp32", False))


class TestFp32BaselineExact(unittest.TestCase):
    def test_cpu_smoke_reproduces_m3_canonical_values(self):
        # The FP32 baseline must be bit-identical after the AMP changes
        # (RMSNorm fp32 compute is a no-op for fp32 inputs; the FP32 step
        # order is untouched). Canonical M3 values from EXP-003.
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = os.path.join(td, "fp32_smoke")
            r = run_train(["--steps", "40", "--t-max", "40",
                           "--ckpt-dir", ckpt_dir, "--log-every", "100",
                           "--ckpt-every", "40", "--data", DATA_BIN])
            self.assertEqual(r.returncode, 0, f"smoke failed: {r.stderr[-500:]}")
            summary = json.load(open(os.path.join(ckpt_dir, "last_summary.json")))
            self.assertEqual(summary["final_train_loss"], 17.793315887451172)
            self.assertEqual(summary["final_val_loss"], 18.236953735351562)
            self.assertEqual(summary["param_hash"], 1200.7129848422692)
            self.assertEqual(summary["precision"], "fp32")
            self.assertEqual(summary["skipped_steps"], 0)
            self.assertIsNone(summary["scaler_final_scale"])
            self.assertIsNotNone(summary["steady_tok_per_s"])

    def test_fp32_and_amp_dirs_are_separate_labels(self):
        # Evidence separation is a directory + label property, not a hope:
        # an fp32 summary must say fp32, so a later AMP summary in another
        # directory can never be mistaken for it.
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = os.path.join(td, "amp_bench", "fp32")
            r = run_train(["--steps", "6", "--t-max", "6",
                           "--ckpt-dir", ckpt_dir, "--log-every", "100",
                           "--ckpt-every", "6", "--data", DATA_BIN])
            self.assertEqual(r.returncode, 0, f"smoke failed: {r.stderr[-500:]}")
            summary = json.load(open(os.path.join(ckpt_dir, "last_summary.json")))
            self.assertEqual(summary["precision"], "fp32")
            self.assertFalse(summary["ddp"])


class TestDdpRank0Writes(unittest.TestCase):
    def test_only_rank0_writes_summary_and_checkpoints(self):
        # CPU/gloo DDP via file rendezvous. Workers are launched directly
        # (NOT via torchrun): torchrun's own worker rendezvous uses a
        # TCPStore, and this VM's loopback TCPStore handshake is corrupted
        # ("Ping failed, invalid value returned" -- see EXP-005), which hangs
        # workers for 900 s. Direct launch with FileStore avoids it entirely.
        # Tiny run; the gate is in the merged output: exactly one SUMMARY
        # line and one checkpoint line, i.e. rank 1 wrote neither.
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = os.path.join(td, "ddp_ckpts")
            init_file = os.path.join(td, "dist_init")
            base_env = dict(os.environ)
            base_env["FORGE_DIST_INIT_METHOD"] = f"file://{init_file}"
            base_env["WORLD_SIZE"] = "2"
            procs = []
            for rank in ("0", "1"):
                env = dict(base_env)
                env["RANK"] = rank
                env["LOCAL_RANK"] = rank
                procs.append(subprocess.Popen(
                    [VENV_PY, TRAIN_PY,
                     "--ddp", "--steps", "4", "--t-max", "4",
                     "--ckpt-dir", ckpt_dir, "--log-every", "100",
                     "--ckpt-every", "4", "--val-every", "100",
                     "--data", DATA_BIN],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=ROOT, env=env))
            outs = []
            try:
                for p in procs:
                    outs.append(p.communicate(timeout=240)[0])
            finally:
                for p in procs:
                    if p.poll() is None:
                        p.kill()
            for p in procs:
                self.assertEqual(p.returncode, 0,
                                 f"ddp worker failed: {''.join(outs)[-800:]}")
            out = "".join(outs)
            self.assertEqual(out.count("SUMMARY"), 1,
                             "expected exactly one SUMMARY line (rank 0 only)")
            self.assertEqual(out.count("checkpoint ->"), 1,
                             "expected exactly one checkpoint line (rank 0 only)")
            summary = json.load(open(os.path.join(ckpt_dir, "last_summary.json")))
            self.assertTrue(summary["ddp"])
            self.assertEqual(summary["world_size"], 2)
            self.assertEqual(summary["steps"], 4)
            self.assertEqual(summary["precision"], "fp32")


if __name__ == "__main__":
    unittest.main(verbosity=2)
