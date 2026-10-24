"""Adversarial tests for eval_harness.py sandbox claims.

These tests DISTINGUISH functional correctness (does the harness score
code correctly?) from sandbox security (can the candidate escape?).

Each test plants a canary and checks whether candidate code can reach it.
A test that REACHES the canary documents a sandbox HOLE (expected to
fail pre-mitigation); a test the harness blocks documents a defense.

Run: python3 benchmarks/test_harness_adversarial.py
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_harness import run_candidate

CANARY = "FORGE_CANARY_9f8e7d6c5b4a"


class TestAdversarial(unittest.TestCase):
    def test_infinite_loop_times_out(self):
        """Harness MUST kill infinite loops (defense expected to hold)."""
        # Module-level loop: completion is appended after the prompt, so a
        # dedented while-loop runs at import time.
        r = run_candidate("x = 1", "\nwhile True:\n    pass",
                          ["assert True"], timeout=5)
        self.assertFalse(r.passed)
        self.assertEqual(r.error_class, "timeout")

    def _attack(self, attack_code: str, timeout: int = 10):
        """Run attack_code at module level; return True iff its asserts passed."""
        r = run_candidate("x = 1", "\n" + attack_code, ["assert True"], timeout=timeout)
        return r.passed

    def test_read_parent_file(self):
        """Can candidate read files OUTSIDE its temp dir? (hole if yes)"""
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(CANARY)
            secret_path = f.name
        try:
            reached = self._attack(
                f"data = open({secret_path!r}).read()\nassert {CANARY!r} in data\n")
            print(f"\n  [audit] read-parent-file: {'HOLE - reached' if reached else 'blocked'}")
        finally:
            os.unlink(secret_path)

    def test_read_environ(self):
        """Can candidate read process environment? (hole if yes)"""
        os.environ["FORGE_TEST_SECRET"] = CANARY
        try:
            reached = self._attack(
                "import os\n"
                f"assert os.environ.get('FORGE_TEST_SECRET') == {CANARY!r}\n")
            print(f"\n  [audit] read-environ: {'HOLE - reached' if reached else 'blocked'}")
        finally:
            del os.environ["FORGE_TEST_SECRET"]

    def test_write_outside_tempdir(self):
        """Can candidate write outside its temp dir? (hole if yes)"""
        with tempfile.TemporaryDirectory() as td:
            target = os.path.join(td, "pwned.txt")
            reached = self._attack(
                f"open({target!r}, 'w').write('x')\n"
                "import os\nassert os.path.exists(" + repr(target) + ")\n")
            print(f"\n  [audit] write-outside-tempdir: {'HOLE - reached' if reached else 'blocked'}")

    def test_spawn_subprocess(self):
        """Can candidate spawn subprocesses? (hole if yes)"""
        reached = self._attack(
            "import subprocess, sys\n"
            "p = subprocess.run([sys.executable, '-c', 'print(1)'], capture_output=True)\n"
            "assert p.returncode == 0\n")
        print(f"\n  [audit] spawn-subprocess: {'HOLE - reached' if reached else 'blocked'}")

    def test_memory_hog_limited(self):
        """2 GiB allocation should die under the 1 GiB rlimit (mitigation check)."""
        r = run_candidate("x = 1", "\nx = bytearray(2 * 1024**3)\nassert len(x) > 0\n",
                          ["assert True"], timeout=15)
        limited = not r.passed
        print(f"\n  [audit] memory-hog-2GiB: {'LIMITED' if limited else 'NOT LIMITED'} "
              f"(class={r.error_class})")


if __name__ == "__main__":
    print("=== Adversarial harness audit (documents holes, not just passes) ===")
    unittest.main(verbosity=2)
