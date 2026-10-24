"""FORGE benchmark decontamination — multi-level detectors.

Levels:
  L1 exact      : sha256(raw bytes) against benchmark doc hashes
  L2 normalized : sha256(normalized text) — lowercase, whitespace and
                  punctuation stripped, comments/strings kept as tokens
  L3 n-gram     : 10-gram containment — fraction of the benchmark doc's
                  distinctive n-grams present in the candidate
  L4 near-dup   : Jaccard >= 0.7 on 5-gram shingles vs benchmark docs
  L5 path       : repo/path contains a known benchmark fingerprint

Limitations (documented, not hidden):
- N-gram methods MISS paraphrased/reworded contamination (Deng et al.
  report F1 ~ 0 on rephrased samples). A model trained on reworded
  HumanEval can score highly while passing every n-gram check.
- Short generic snippets (e.g. `def add(a, b): return a + b`) produce
  false positives; L3 uses containment thresholds and minimum doc
  sizes to limit this, and every hit is logged for human review.
- Unknown benchmarks cannot be checked. "Zero hits" means "zero hits
  against THIS fingerprint set", never "provably clean".

The fingerprint set lives in benchmarks/fingerprints.json and is built
by benchmarks/build_fingerprints.py from public benchmark releases.
It is QUARANTINED: never used as training data, never tuned against.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata

from data_pipeline import jaccard, shingles, PATH_FINGERPRINTS


def normalize_code(text: str) -> str:
    """Aggressive normalization for L2 matching."""
    text = unicodedata.normalize("NFKC", text).lower()
    # strip comments for python (# ...) but keep code
    lines = []
    for line in text.split("\n"):
        lines.append(line.split("#")[0])
    text = "\n".join(lines)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text


def ngrams(tokens: list, n: int) -> set:
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


_WORD = re.compile(r"[a-z_][a-z0-9_]*|[0-9]+")


def load_fingerprints(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


class Decontaminator:
    """Applies L1..L5 checks. Records hits per record; excludes on policy."""

    def __init__(self, fingerprints: dict, n: int = 10,
                 containment_threshold: float = 0.5,
                 jaccard_threshold: float = 0.7,
                 exclude_on_hit: bool = True):
        self.fp = fingerprints
        self.n = n
        self.containment_threshold = containment_threshold
        self.jaccard_threshold = jaccard_threshold
        self.exclude_on_hit = exclude_on_hit
        # Precompute per-benchmark-doc n-gram sets and shingle sets
        self._doc_ngrams = {}
        self._doc_shingles = {}
        self._doc_exact = set()
        self._doc_norm = set()
        for doc in fingerprints.get("docs", []):
            did = doc["id"]
            text = doc["text"]
            self._doc_exact.add(hashlib.sha256(
                text.encode("utf-8")).hexdigest())
            self._doc_norm.add(hashlib.sha256(
                normalize_code(text).encode("utf-8")).hexdigest())
            toks = _WORD.findall(text.lower())
            self._doc_ngrams[did] = ngrams(toks, n) if len(toks) >= n else set()
            self._doc_shingles[did] = shingles(text, 1500)

    def check_record(self, rec: dict) -> list:
        """Return list of (level, benchmark_id) hits. Pure function."""
        hits = []
        content = rec["content"]
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if h in self._doc_exact:
            hits.append(("L1_exact", "benchmark_doc"))
        nh = hashlib.sha256(normalize_code(content).encode("utf-8")).hexdigest()
        if nh in self._doc_norm:
            hits.append(("L2_normalized", "benchmark_doc"))
        toks = _WORD.findall(content.lower())
        cand_ng = ngrams(toks, self.n) if len(toks) >= self.n else set()
        cand_sh = shingles(content, 1500)
        for doc in self.fp.get("docs", []):
            did = doc["id"]
            bng = self._doc_ngrams[did]
            if bng and cand_ng:
                containment = len(bng & cand_ng) / len(bng)
                if containment >= self.containment_threshold:
                    hits.append(("L3_ngram", did))
            bsh = self._doc_shingles[did]
            if bsh and cand_sh and jaccard(bsh, cand_sh) >= self.jaccard_threshold:
                hits.append(("L4_neardup", did))
        blob = f"{rec.get('repo','')} {rec.get('path','')}".lower()
        for marker in PATH_FINGERPRINTS:
            if marker in blob:
                hits.append(("L5_path", marker))
                break
        return hits

    def apply(self, records: list, stats: dict) -> list:
        for rec in records:
            hits = self.check_record(rec)
            if hits:
                levels = sorted({lv for lv, _ in hits})
                rec["decontam_hits"] = [{"level": lv, "doc": d} for lv, d in hits]
                stats["decontam_flagged"] = stats.get("decontam_flagged", 0) + 1
                for lv in levels:
                    stats[f"decontam_{lv}"] = stats.get(f"decontam_{lv}", 0) + 1
                if self.exclude_on_hit:
                    rec["excluded"] = True
                    rec["exclusion_reason"] = "benchmark_overlap:" + ",".join(levels)
                    stats["excluded_benchmark_overlap"] = stats.get(
                        "excluded_benchmark_overlap", 0) + 1
        return records
