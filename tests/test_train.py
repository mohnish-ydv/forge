"""Gate D: training-loop validation tests. Requires torch (run with .venv python).

Covers the 8 required checks:
 1. loss finite on valid input
 2. output/target shapes agree
 3. backward gives finite gradients
 4. parameters update after optimizer step
 5. checkpoint save/load reproduces eval-mode outputs
 6. validation does not update parameters
 7. loss decreases on a tiny memorizable dataset
 8. fresh-process resume completes (subprocess)
"""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from model import ForgeModel, ForgeConfig
from train import (save_checkpoint, load_checkpoint, make_batches,
                   evaluate, param_hash)


def tiny_model():
    cfg = ForgeConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                      n_kv_heads=2, max_seq_len=33)
    torch.manual_seed(7)
    return ForgeModel(cfg), cfg


def tiny_batch():
    torch.manual_seed(11)
    return torch.randint(0, 256, (4, 32))


class TestTrainLoop(unittest.TestCase):
    def test_1_loss_finite(self):
        m, _ = tiny_model()
        x = tiny_batch()
        _, loss = m(x[:, :-1], x[:, 1:])
        self.assertTrue(torch.isfinite(loss).item())

    def test_2_shapes_agree(self):
        m, cfg = tiny_model()
        x = tiny_batch()
        logits, loss = m(x[:, :-1], x[:, 1:])
        self.assertEqual(logits.shape, (4, 31, cfg.vocab_size))
        self.assertEqual(loss.shape, torch.Size([]))

    def test_3_grads_finite(self):
        m, _ = tiny_model()
        x = tiny_batch()
        _, loss = m(x[:, :-1], x[:, 1:])
        m.zero_grad()
        loss.backward()
        for p in m.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all().item())

    def test_4_params_update(self):
        m, _ = tiny_model()
        x = tiny_batch()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        before = [p.detach().clone() for p in m.parameters()]
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        changed = any(not torch.equal(a, b)
                      for a, b in zip(before, m.parameters()))
        self.assertTrue(changed, "no parameter changed after optimizer step")

    def test_5_checkpoint_reproduces_outputs(self):
        m, _ = tiny_model()
        x = tiny_batch()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ckpt.pt")
            save_checkpoint(path, m, opt, None, 1, 128, {"t": 1})
            m2, _ = tiny_model()  # different init (seed 7 vs fresh below)
            torch.manual_seed(999)
            m2 = ForgeModel(ForgeConfig(vocab_size=256, d_model=64, n_layers=2,
                                        n_heads=4, n_kv_heads=2, max_seq_len=33))
            opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
            load_checkpoint(path, m2, opt2, None)
            m.eval()
            m2.eval()
            with torch.no_grad():
                o1, _ = m(x[:, :-1])
                o2, _ = m2(x[:, :-1])
            self.assertTrue(torch.equal(o1, o2),
                            "eval outputs differ after checkpoint round-trip")

    def test_6_val_does_not_update_params(self):
        m, _ = tiny_model()
        batches = [tiny_batch() for _ in range(2)]
        before = param_hash(m)
        evaluate(m, batches, torch.device("cpu"))
        self.assertEqual(before, param_hash(m))

    def test_6b_param_hash_matches_hand_computed_sum(self):
        # Regression: param_hash must equal the plain float64 sum of all params,
        # computed with the accumulator on CPU (see Kaggle GPU smoke run, where a
        # CUDA param sum added to a CPU accumulator raised at SUMMARY time).
        m, _ = tiny_model()
        expected = 0.0
        for p in m.parameters():
            expected += float(p.detach().double().sum().cpu().item())
        self.assertEqual(param_hash(m), expected)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_6c_param_hash_on_cuda_model(self):
        # The exact Kaggle failure: model on cuda:0, accumulator must not mix
        # devices. Fails on the old code, passes with the .cpu() fix.
        m, _ = tiny_model()
        m = m.to("cuda")
        expected = 0.0
        for p in m.parameters():
            expected += float(p.detach().double().sum().cpu().item())
        self.assertEqual(param_hash(m), expected)

    def test_7_loss_decreases_on_memorizable(self):
        # Tiny repetitive data the model can memorize: loss must fall.
        torch.manual_seed(3)
        m, _ = tiny_model()
        base = torch.randint(0, 256, (32,))  # one 32-token pattern (<= max_seq_len 33)
        data = base.repeat(8, 1)  # 8 identical rows -> memorizable
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        losses = []
        m.train()
        for _ in range(30):
            _, loss = m(data[:, :-1], data[:, 1:])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * 0.5,
                        f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}")

    def test_8_resume_fresh_process(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        venv_py = os.path.join(root, ".venv", "bin", "python")
        train_py = os.path.join(root, "src", "train.py")
        data_bin = os.path.join(root, "data", "toy", "train.bin")
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = os.path.join(td, "ckpts")
            env = dict(os.environ)
            # phase 1: 6 steps
            r1 = subprocess.run(
                [venv_py, train_py, "--steps", "6", "--t-max", "12",
                 "--ckpt-every", "6", "--ckpt-dir", ckpt_dir,
                 "--data", data_bin, "--log-every", "100"],
                capture_output=True, text=True, timeout=300, cwd=root, env=env)
            self.assertEqual(r1.returncode, 0, f"phase1 failed: {r1.stderr[-500:]}")
            # phase 2: fresh process resumes to 12
            r2 = subprocess.run(
                [venv_py, train_py, "--resume", "--steps", "12", "--t-max", "12",
                 "--ckpt-every", "6", "--ckpt-dir", ckpt_dir,
                 "--data", data_bin, "--log-every", "100"],
                capture_output=True, text=True, timeout=300, cwd=root, env=env)
            self.assertEqual(r2.returncode, 0, f"phase2 failed: {r2.stderr[-500:]}")
            self.assertIn("resumed from", r2.stdout)
            self.assertIn("SUMMARY", r2.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
