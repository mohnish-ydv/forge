"""Unit tests for multi-level benchmark decontamination. Stdlib unittest."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decontam import Decontaminator, normalize_code
from data_pipeline import new_record


BENCH_TEXT = (
    "def has_close_elements(numbers, threshold):\n"
    '    """Check if any two numbers are closer than threshold."""\n'
    "    for i in range(len(numbers)):\n"
    "        for j in range(i + 1, len(numbers)):\n"
    "            if abs(numbers[i] - numbers[j]) < threshold:\n"
    "                return True\n"
    "    return False\n"
)
FP = {"docs": [{"id": "HumanEval/1", "source": "test", "text": BENCH_TEXT}]}


def rec(content, **kw):
    kw.setdefault("language", "python")
    kw.setdefault("license", "mit")
    return new_record(0, content, **kw)


class TestDecontam(unittest.TestCase):
    def setUp(self):
        self.d = Decontaminator(FP)

    def test_l1_exact(self):
        hits = self.d.check_record(rec(BENCH_TEXT))
        self.assertIn(("L1_exact", "benchmark_doc"), hits)

    def test_l2_normalized(self):
        # Same code, different case/whitespace/comments -> L2 catches it
        variant = BENCH_TEXT.upper().replace("    ", "\t") + "# a comment\n"
        hits = self.d.check_record(rec(variant))
        levels = {lv for lv, _ in hits}
        self.assertIn("L2_normalized", levels)

    def test_l5_path(self):
        hits = self.d.check_record(
            rec("x = 1\n" * 30, repo="someone/humaneval-solutions", path="sol.py"))
        self.assertIn(("L5_path", "humaneval"), hits)

    def test_clean_passes(self):
        code = ("def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n"
                "        a, b = b, a + b\n    return a\n")
        self.assertEqual(self.d.check_record(rec(code)), [])

    def test_apply_excludes(self):
        from collections import defaultdict
        stats = defaultdict(int)
        records = [rec(BENCH_TEXT), rec("x = 1\n" * 30)]
        out = self.d.apply(records, stats)
        self.assertTrue(records[0]["excluded"])
        self.assertIn("benchmark_overlap", records[0]["exclusion_reason"])
        self.assertFalse(records[1]["excluded"])
        self.assertEqual(len(out), 2)  # apply marks; pipeline filters later

    def test_normalize_deterministic(self):
        self.assertEqual(normalize_code("Ab C!"), normalize_code("ab c!"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
