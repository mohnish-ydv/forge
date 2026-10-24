"""M4 regression tests: benchmark integrity, scoring, contamination, safety.

Covers Gate A (evaluator correctness), Gate B (held-out integrity +
contamination controls), Gate C (execution-safety boundaries), Gate D
(baselines). Slow tests (full oracle sweep) are included — the whole file
runs in well under a minute.
"""
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "src"))

from eval_harness import Problem, evaluate, run_candidate  # noqa: E402
from decontam import Decontaminator  # noqa: E402

BENCH_PATH = ROOT / "benchmarks" / "forge_heldout_v1.json"
FPS_PATH = ROOT / "benchmarks" / "fingerprints.json"


def load_bench():
    return json.loads(BENCH_PATH.read_text())


def problems_of(bench, split="all"):
    tasks = [t for t in bench["tasks"] if split == "all" or t["split"] == split]
    return [Problem(name=t["id"], prompt=t["prompt"], tests=t["tests"],
                    entry_point=t["entry_point"]) for t in tasks]


class TestBenchmarkIntegrity(unittest.TestCase):
    def test_version_splits_and_freeze(self):
        b = load_bench()
        self.assertEqual(b["version"], "forge-heldout-v1")
        self.assertTrue(b["frozen"])
        self.assertEqual(len(b["tasks"]), 24)
        dev = [t for t in b["tasks"] if t["split"] == "dev"]
        held = [t for t in b["tasks"] if t["split"] == "heldout"]
        self.assertEqual(len(dev), 12)
        self.assertEqual(len(held), 12)
        self.assertEqual(len({t["id"] for t in b["tasks"]}), 24)

    def test_hashes_verify(self):
        b = load_bench()
        digest = b.pop("benchmark_sha256")
        canon = json.dumps(b, sort_keys=True, separators=(",", ":"))
        self.assertEqual(hashlib.sha256(canon.encode()).hexdigest(), digest)
        for t in b["tasks"]:
            self.assertEqual(hashlib.sha256(t["prompt"].encode()).hexdigest(),
                             t["prompt_sha256"])
            self.assertEqual(
                hashlib.sha256("\n".join(t["tests"]).encode()).hexdigest(),
                t["tests_sha256"])

    def test_every_task_has_tests_and_entry_point(self):
        for t in load_bench()["tasks"]:
            self.assertGreaterEqual(len(t["tests"]), 3, t["id"])
            self.assertIn(t["entry_point"], t["prompt"], t["id"])


class TestScoring(unittest.TestCase):
    def test_oracle_scores_perfect(self):
        sys.path.insert(0, str(ROOT / "benchmarks"))
        import build_heldout
        b = load_bench()
        comps = {t["id"]: build_heldout.ORACLES[t["id"]] for t in b["tasks"]}
        r = evaluate(problems_of(b), comps)
        self.assertEqual(r["pass_at_1"], 1.0, r["error_funnel"])

    def test_empty_baseline_scores_zero(self):
        b = load_bench()
        comps = {t["id"]: "    pass" for t in b["tasks"]}
        r = evaluate(problems_of(b), comps)
        self.assertEqual(r["pass_at_1"], 0.0)

    def test_none_baseline_scores_zero(self):
        b = load_bench()
        comps = {t["id"]: "    return None" for t in b["tasks"]}
        r = evaluate(problems_of(b), comps)
        self.assertEqual(r["pass_at_1"], 0.0)

    def test_wrong_answer_is_assertion_not_pass(self):
        r = run_candidate("def add(a, b):\n", "    return a - b\n",
                          ["assert add(2, 3) == 5"])
        self.assertFalse(r.passed)
        self.assertEqual(r.error_class, "assertion")

    def test_exception_is_not_pass(self):
        r = run_candidate("def f():\n", "    return 1 // 0\n", ["assert f() == 1"])
        self.assertFalse(r.passed)
        self.assertEqual(r.error_class, "import/runtime")

    def test_syntax_error_is_not_pass(self):
        r = run_candidate("def f():\n", "    def broken(:\n", ["assert True"])
        self.assertFalse(r.passed)
        self.assertEqual(r.error_class, "syntax")

    def test_infinite_loop_times_out(self):
        r = run_candidate("", "while True:\n    pass\n", ["assert True"], timeout=3)
        self.assertFalse(r.passed)
        self.assertEqual(r.error_class, "timeout")

    def test_fixed_marker_spoof_does_not_pass(self):
        # Old fixed marker + early exit must NOT score (per-run token defeats it).
        r = run_candidate("def f():\n    pass\n",
                          "print('__FORGE_PASS__')\nimport sys; sys.exit(0)\n",
                          ["assert False, 'never reached'"])
        self.assertFalse(r.passed)

    def test_empty_completion_is_not_pass(self):
        r = run_candidate("def f():\n", "", ["assert f() == 1"])
        self.assertFalse(r.passed)


class TestContaminationControls(unittest.TestCase):
    def test_heldout_clean_vs_benchmark_fingerprints(self):
        fps = json.loads(FPS_PATH.read_text())
        dec = Decontaminator(fps, exclude_on_hit=False)
        for t in load_bench()["tasks"]:
            hits = dec.check_record(
                {"id": t["id"], "content": t["prompt"] + "\n" + "\n".join(t["tests"])})
            self.assertEqual(hits, [], f"contamination hit on {t['id']}: {hits}")

    def test_positive_control_verbatim_is_caught(self):
        fps = json.loads(FPS_PATH.read_text())
        dec = Decontaminator(fps, exclude_on_hit=False)
        hits = dec.check_record({"id": "control", "content": fps["docs"][0]["text"]})
        levels = {h[0] for h in hits}
        self.assertTrue({"L1_exact", "L2_normalized", "L3_ngram"} & levels,
                        f"verbatim doc not caught: {hits}")

    def test_heldout_clean_vs_m2_corpus(self):
        from decontam import normalize_code
        corpus_hashes = set()
        with (ROOT / "data" / "m2_a" / "corpus.jsonl").open() as fh:
            for line in fh:
                c = json.loads(line)["content"]
                corpus_hashes.add(hashlib.sha256(c.encode()).hexdigest())
                corpus_hashes.add("N" + hashlib.sha256(normalize_code(c).encode()).hexdigest())
        self.assertGreater(len(corpus_hashes), 4000)
        for t in load_bench()["tasks"]:
            text = t["prompt"] + "\n" + "\n".join(t["tests"])
            h1 = hashlib.sha256(text.encode()).hexdigest()
            h2 = "N" + hashlib.sha256(normalize_code(text).encode()).hexdigest()
            self.assertNotIn(h1, corpus_hashes, t["id"])
            self.assertNotIn(h2, corpus_hashes, t["id"])


class TestSafetyBoundaries(unittest.TestCase):
    def test_sandbox_mode_recorded(self):
        r = run_candidate("def f():\n", "    return 1\n", ["assert f() == 1"])
        self.assertTrue(r.passed)
        self.assertIn(r.sandbox, ("pivot", "plain"))

    def test_host_files_unreachable_in_pivot(self):
        r = run_candidate("def f():\n", "    return 1\n", ["assert f() == 1"])
        if r.sandbox != "pivot":
            self.skipTest("pivot sandbox unavailable on this machine")
        probe = run_candidate("", "open('/home/hatch/USER.md').read()\n", ["assert True"])
        self.assertFalse(probe.passed)
        self.assertEqual(probe.error_class, "import/runtime")

    def test_oldroot_masked_in_pivot(self):
        r = run_candidate("def f():\n", "    return 1\n", ["assert f() == 1"])
        if r.sandbox != "pivot":
            self.skipTest("pivot sandbox unavailable on this machine")
        probe = run_candidate("", "open('/oldroot/etc/hostname').read()\n", ["assert True"])
        self.assertFalse(probe.passed)

    def test_network_unreachable_in_pivot(self):
        r = run_candidate("def f():\n", "    return 1\n", ["assert f() == 1"])
        if r.sandbox != "pivot":
            self.skipTest("pivot sandbox unavailable on this machine")
        probe = run_candidate(
            "", "import socket; socket.create_connection(('8.8.8.8', 53), timeout=2)\n",
            ["assert True"])
        self.assertFalse(probe.passed)

    def test_cleared_environment(self):
        import os
        os.environ["FORGE_TEST_SECRET"] = "canary-123"
        try:
            probe = run_candidate(
                "", "import os\nassert os.environ.get('FORGE_TEST_SECRET') == 'canary-123'\n",
                ["assert True"])
            self.assertFalse(probe.passed, "candidate inherited host environment!")
        finally:
            del os.environ["FORGE_TEST_SECRET"]

    def test_fork_bomb_contained(self):
        probe = run_candidate("", "import os\nwhile True:\n    os.fork()\n",
                              ["assert True"], timeout=15)
        self.assertFalse(probe.passed)


if __name__ == "__main__":
    unittest.main(verbosity=1)
