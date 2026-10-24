"""Unit tests for the M2 data pipeline. Stdlib unittest — runs on VM, Kaggle, Termux."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_pipeline import (
    PipelineConfig, new_record, prefilter, license_ok, quality_check,
    scan_secrets, exact_dedup, near_dedup, assign_split, jaccard,
    shingles, write_manifest,
)
from collections import defaultdict


def rec(content, **kw):
    kw.setdefault("language", "python")
    kw.setdefault("license", "mit")
    return new_record(0, content, **kw)


class TestPrefilter(unittest.TestCase):
    def setUp(self):
        self.cfg = PipelineConfig(min_bytes=10, max_bytes=10000)

    def test_language(self):
        self.assertEqual(prefilter(rec("x = 1", language="java"), self.cfg), "language")

    def test_empty(self):
        self.assertEqual(prefilter(rec("   \n  "), self.cfg), "empty")

    def test_too_small(self):
        self.assertEqual(prefilter(rec("x=1"), self.cfg), "too_small")

    def test_too_large(self):
        cfg = PipelineConfig(max_bytes=20)
        self.assertEqual(prefilter(rec("x = " + "1" * 100), cfg), "too_large")

    def test_binary(self):
        self.assertEqual(prefilter(rec("abc\x00def" * 10), self.cfg), "binary")

    def test_generated(self):
        self.assertEqual(
            prefilter(rec("# This file is auto-generated. Do not edit.\nx=1" * 5), self.cfg),
            "generated")

    def test_minified(self):
        cfg = PipelineConfig(max_line_len=50, min_bytes=10)
        self.assertEqual(prefilter(rec("x=" + "1," * 100), cfg), "minified")

    def test_pass(self):
        self.assertEqual(prefilter(rec("def f(a):\n    return a + 1\n"), self.cfg), "")


class TestLicense(unittest.TestCase):
    def test_permissive(self):
        cfg = PipelineConfig()
        for lic in ("mit", "apache-2.0", "bsd-3-clause", "isc"):
            self.assertTrue(license_ok(rec("x=1", license=lic), cfg), lic)

    def test_blocked(self):
        cfg = PipelineConfig()
        for lic in ("gpl-3.0", "no_license", "", "agpl-3.0"):
            self.assertFalse(license_ok(rec("x=1", license=lic), cfg), lic)


class TestQuality(unittest.TestCase):
    def test_unparseable(self):
        cfg = PipelineConfig(drop_unparseable=True)
        self.assertEqual(quality_check(rec("def f(:\n  broken ((("), cfg), "unparseable")

    def test_unparseable_kept_when_disabled(self):
        cfg = PipelineConfig(drop_unparseable=False)
        self.assertEqual(quality_check(rec("def f(:\n  broken ((("), cfg), "")

    def test_low_alpha(self):
        # Parseable but nearly no letters -> low_alpha_frac (not unparseable)
        cfg = PipelineConfig(min_alpha_frac=0.5)
        code = "x = 1234567890123456789012345678901234567890\n" * 5
        self.assertEqual(quality_check(rec(code), cfg), "low_alpha_frac")

    def test_unusual_code_preserved(self):
        # Unusual but legitimate: unicode identifiers, decorators, metaclasses
        code = ("# -*- coding: utf-8 -*-\nimport functools\n\n"
                "def décor(f):\n    @functools.wraps(f)\n    def w(*a, **k):\n        return f(*a, **k)\n    return w\n")
        cfg = PipelineConfig()
        self.assertEqual(quality_check(rec(code), cfg), "")
        self.assertEqual(prefilter(rec(code), cfg), "")


class TestSecrets(unittest.TestCase):
    def test_private_key(self):
        hits = scan_secrets(rec("k = '-----BEGIN RSA PRIVATE KEY-----\nabc'"))
        self.assertIn("private_key", hits)

    def test_aws_key(self):
        hits = scan_secrets(rec("x = 'AKIAIOSFODNN7EXAMPLE'"))
        self.assertIn("aws_key", hits)

    def test_github_token(self):
        hits = scan_secrets(rec("t = 'ghp_abcdefghijklmnopqrst1234'"))
        self.assertIn("github_token", hits)

    def test_password_assign(self):
        hits = scan_secrets(rec("password = 's3cr3t-value'"))
        self.assertIn("generic_secret_assign", hits)

    def test_clean(self):
        code = "def login(user, password):\n    return authenticate(user, password)\n"
        self.assertEqual(scan_secrets(rec(code)), [])


class TestExactDedup(unittest.TestCase):
    def test_dup_removed_keeps_first(self):
        a = new_record(1, "x = 1\n", language="python", license="mit")
        b = new_record(2, "x = 1\n", language="python", license="mit")
        c = new_record(3, "y = 2\n", language="python", license="mit")
        stats = defaultdict(int)
        kept = exact_dedup([a, b, c], stats)
        self.assertEqual([r["id"] for r in kept], ["doc-000001", "doc-000003"])
        self.assertTrue(b["excluded"] and "exact_dup_of" in b["exclusion_reason"])
        self.assertEqual(stats["exact_dups"], 1)


class TestNearDedup(unittest.TestCase):
    def _mk(self, i, content):
        return new_record(i, content, language="python", license="mit")

    def test_near_dup_pair(self):
        # Realistic-size doc: small edit keeps Jaccard high (verified 0.853).
        # Tiny 6-line functions are NOT reliably caught at 0.7 — documented
        # limitation (a single added line changes most 5-grams).
        lines = ["def process_records(records):", "    results = []"]
        for i in range(30):
            lines.append(f"    value_{i} = records.get('field_{i}', 0) * {i + 1}")
        lines += ["    return results"]
        base = "\n".join(lines) + "\n"
        vlines = list(lines)
        vlines[5] = "    value_4 = records.get('field_4', 0) * 999  # tuned constant"
        vlines.insert(10, "    # validated against schema v2")
        variant = "\n".join(vlines) + "\n"
        other = "def completely_different(zebra, qux):\n    return zebra ** qux - 42\n"
        cfg = PipelineConfig(near_dup_jaccard=0.7)
        stats = defaultdict(int)
        kept = near_dedup([self._mk(1, base), self._mk(2, variant), self._mk(3, other)], cfg, stats)
        kept_ids = [r["id"] for r in kept]
        self.assertIn("doc-000001", kept_ids)
        self.assertIn("doc-000003", kept_ids)
        self.assertNotIn("doc-000002", kept_ids)
        self.assertEqual(stats["near_dups"], 1)

    def test_distinct_kept(self):
        docs = [self._mk(i, f"def unique_function_{i}(x):\n    return x * {i} + {i*7}\n") for i in range(5)]
        cfg = PipelineConfig()
        stats = defaultdict(int)
        kept = near_dedup(docs, cfg, stats)
        self.assertEqual(len(kept), 5)


class TestSplit(unittest.TestCase):
    def test_deterministic(self):
        cfg = PipelineConfig()
        r = new_record(1, "x = 1\n")
        self.assertEqual(assign_split(r, cfg), assign_split(r, cfg))

    def test_distribution_sums(self):
        cfg = PipelineConfig(train_frac=0.9, val_frac=0.05)
        counts = defaultdict(int)
        for i in range(2000):
            r = new_record(i, f"content-{i}\n" + "x" * 50)
            counts[assign_split(r, cfg)] += 1
        self.assertEqual(sum(counts.values()), 2000)
        # hash-based split should be roughly proportional (loose bounds)
        self.assertGreater(counts["train"], 1600)
        self.assertLess(counts["train"], 2000)

    def test_dup_same_split(self):
        # identical content -> identical sha -> identical split (dup-proof)
        cfg = PipelineConfig()
        a = new_record(1, "same content here\n" + "y" * 100)
        b = new_record(2, "same content here\n" + "y" * 100)
        self.assertEqual(assign_split(a, cfg), assign_split(b, cfg))


class TestManifest(unittest.TestCase):
    def test_no_content_leak(self):
        r = new_record(1, "secret-code-body", path="a.py", repo="r",
                       language="python", license="mit")
        r["split"] = "train"
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            write_manifest([r], path, {"source": "test"})
            with open(path) as f:
                entry = json.loads(f.readline())
            self.assertNotIn("content", entry)
            self.assertEqual(entry["sha256"], r["sha256"])
            self.assertEqual(entry["provenance"]["source"], "test")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
