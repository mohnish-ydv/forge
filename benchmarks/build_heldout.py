"""FORGE-Heldout v1 builder.

Authors 24 ORIGINAL coding tasks (12 dev + 12 held-out), verifies them
against reference solutions, and writes benchmarks/forge_heldout_v1.json
with per-task hashes, provenance, and a frozen version tag.

All tasks and tests were written fresh for FORGE (no copying from public
benchmarks). Dedicated CC0-1.0.

QUARANTINE: this file contains reference (oracle) solutions. It must NEVER
be included in training data or model-facing prompts. See quarantine/README.md.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_harness import Problem, run_candidate  # noqa: E402

VERSION = "forge-heldout-v1"
CREATED = "2026-10-24"


def task(task_id: str, split: str, prompt: str, tests: list[str], entry_point: str):
    return {
        "id": task_id,
        "split": split,
        "prompt": prompt,
        "tests": tests,
        "entry_point": entry_point,
    }


TASKS = [
    # ---------------- DEV split (12) — may be used for iteration ----------------
    task("dev_sum_of_squares", "dev",
         'def sum_of_squares(n):\n    """Return 1^2 + 2^2 + ... + n^2. For n <= 0 return 0."""',
         ["assert sum_of_squares(3) == 14", "assert sum_of_squares(1) == 1",
          "assert sum_of_squares(0) == 0", "assert sum_of_squares(10) == 385",
          "assert sum_of_squares(-5) == 0"], "sum_of_squares"),
    task("dev_is_palindrome", "dev",
         'def is_palindrome(s):\n    """Return True if s reads the same forwards and backwards,\n    ignoring case and non-alphanumeric characters."""',
         ["assert is_palindrome('racecar') is True", "assert is_palindrome('hello') is False",
          "assert is_palindrome('A man, a plan, a canal: Panama') is True",
          "assert is_palindrome('') is True", "assert is_palindrome('a') is True"], "is_palindrome"),
    task("dev_word_count", "dev",
         'def word_count(s):\n    """Return the number of whitespace-separated words in s."""',
         ["assert word_count('hello world') == 2", "assert word_count('') == 0",
          "assert word_count('  a  b  ') == 2", "assert word_count('one') == 1"], "word_count"),
    task("dev_fibonacci_list", "dev",
         'def fibonacci_list(n):\n    """Return a list of the first n Fibonacci numbers, starting 0, 1."""',
         ["assert fibonacci_list(0) == []", "assert fibonacci_list(1) == [0]",
          "assert fibonacci_list(2) == [0, 1]", "assert fibonacci_list(7) == [0, 1, 1, 2, 3, 5, 8]"],
         "fibonacci_list"),
    task("dev_merge_sorted", "dev",
         'def merge_sorted(a, b):\n    """Merge two ascending-sorted lists into one ascending-sorted list."""',
         ["assert merge_sorted([1, 3, 5], [2, 4]) == [1, 2, 3, 4, 5]",
          "assert merge_sorted([], [1]) == [1]", "assert merge_sorted([2], [1]) == [1, 2]",
          "assert merge_sorted([1, 1], [1]) == [1, 1, 1]", "assert merge_sorted([], []) == []"],
         "merge_sorted"),
    task("dev_char_frequency", "dev",
         'def char_frequency(s):\n    """Return a dict mapping each character in s to its count."""',
         ["assert char_frequency('aab') == {'a': 2, 'b': 1}",
          "assert char_frequency('') == {}", "assert char_frequency('Aa') == {'A': 1, 'a': 1}"],
         "char_frequency"),
    task("dev_is_prime", "dev",
         'def is_prime(n):\n    """Return True if n is a prime number, else False."""',
         ["assert is_prime(2) is True", "assert is_prime(1) is False",
          "assert is_prime(0) is False", "assert is_prime(17) is True",
          "assert is_prime(15) is False", "assert is_prime(97) is True"], "is_prime"),
    task("dev_flatten_once", "dev",
         'def flatten_once(nested):\n    """Flatten one level of nesting: [[1,2],[3]] -> [1,2,3].\n    Non-list elements are kept as-is."""',
         ["assert flatten_once([[1, 2], [3]]) == [1, 2, 3]", "assert flatten_once([]) == []",
          "assert flatten_once([[], [1]]) == [1]", "assert flatten_once([1, [2]]) == [1, 2]"],
         "flatten_once"),
    task("dev_title_case", "dev",
         'def title_case(s):\n    """Capitalize the first letter of each word, lowercasing the rest."""',
         ["assert title_case('hello world') == 'Hello World'", "assert title_case('') == ''",
          "assert title_case('a b c') == 'A B C'", "assert title_case('HELLO') == 'Hello'"],
         "title_case"),
    task("dev_gcd", "dev",
         'def gcd(a, b):\n    """Return the greatest common divisor of a and b (non-negative ints)."""',
         ["assert gcd(12, 8) == 4", "assert gcd(0, 5) == 5",
          "assert gcd(7, 7) == 7", "assert gcd(100, 35) == 5"], "gcd"),
    task("dev_unique_sorted", "dev",
         'def unique_sorted(lst):\n    """Return the distinct elements of lst in ascending order."""',
         ["assert unique_sorted([3, 1, 2, 1]) == [1, 2, 3]", "assert unique_sorted([]) == []",
          "assert unique_sorted([5, 5, 5]) == [5]"], "unique_sorted"),
    task("dev_caesar_shift", "dev",
         'def caesar_shift(s, k):\n    """Shift each lowercase letter in s forward by k positions (wrap around).\n    Leave other characters unchanged."""',
         ["assert caesar_shift('abc', 1) == 'bcd'", "assert caesar_shift('xyz', 3) == 'abc'",
          "assert caesar_shift('hello', 0) == 'hello'", "assert caesar_shift('a', 26) == 'a'",
          "assert caesar_shift('a b!', 1) == 'b c!'"], "caesar_shift"),

    # ---------------- HELD-OUT split (12) — never iterate on these ----------------
    task("heldout_run_length_encode", "heldout",
         'def run_length_encode(s):\n    """Encode s as character counts: "aabbbc" -> "a2b3c1".\n    Empty string encodes to ""."""',
         ["assert run_length_encode('aabbbc') == 'a2b3c1'", "assert run_length_encode('') == ''",
          "assert run_length_encode('a') == 'a1'", "assert run_length_encode('abcd') == 'a1b1c1d1'"],
         "run_length_encode"),
    task("heldout_balanced_brackets", "heldout",
         'def balanced_brackets(s):\n    """Return True if every () [] {} in s is correctly matched and nested."""',
         ["assert balanced_brackets('()[]{}') is True", "assert balanced_brackets('([)]') is False",
          "assert balanced_brackets('') is True", "assert balanced_brackets('(((') is False",
          "assert balanced_brackets('{[()]}') is True"], "balanced_brackets"),
    task("heldout_second_largest", "heldout",
         'def second_largest(lst):\n    """Return the second largest DISTINCT value in lst,\n    or None if fewer than two distinct values exist."""',
         ["assert second_largest([1, 2, 3]) == 2", "assert second_largest([10, 10, 9]) == 9",
          "assert second_largest([7]) is None", "assert second_largest([4, 4, 4]) is None",
          "assert second_largest([5, 1, 5, 3]) == 3"], "second_largest"),
    task("heldout_group_anagrams", "heldout",
         'def group_anagrams(words):\n    """Group anagrams together. Return a list of groups; sort each group\n    alphabetically and sort the groups by their first element."""',
         ["assert group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat']) == "
          "[['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]",
          "assert group_anagrams([]) == []", "assert group_anagrams(['a']) == [['a']]"],
         "group_anagrams"),
    task("heldout_pascal_row", "heldout",
         'def pascal_row(n):\n    """Return row n (0-indexed) of Pascal\'s triangle as a list."""',
         ["assert pascal_row(0) == [1]", "assert pascal_row(1) == [1, 1]",
          "assert pascal_row(4) == [1, 4, 6, 4, 1]"], "pascal_row"),
    task("heldout_longest_common_prefix", "heldout",
         'def longest_common_prefix(strs):\n    """Return the longest common prefix of all strings in strs.\n    Empty list -> ""."""',
         ["assert longest_common_prefix(['flower', 'flow', 'flight']) == 'fl'",
          "assert longest_common_prefix([]) == ''",
          "assert longest_common_prefix(['dog']) == 'dog'",
          "assert longest_common_prefix(['a', 'b']) == ''"], "longest_common_prefix"),
    task("heldout_sieve", "heldout",
         'def sieve(n):\n    """Return all prime numbers strictly below n, in ascending order."""',
         ["assert sieve(10) == [2, 3, 5, 7]", "assert sieve(2) == []",
          "assert sieve(20) == [2, 3, 5, 7, 11, 13, 17, 19]", "assert sieve(3) == [2]"],
         "sieve"),
    task("heldout_chunk_list", "heldout",
         'def chunk_list(lst, k):\n    """Split lst into consecutive chunks of size k\n    (the last chunk may be smaller)."""',
         ["assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]",
          "assert chunk_list([], 3) == []", "assert chunk_list([1, 2], 5) == [[1, 2]]"],
         "chunk_list"),
    task("heldout_most_frequent", "heldout",
         'def most_frequent(lst):\n    """Return the most frequent element of lst.\n    Ties are broken by first appearance."""',
         ["assert most_frequent([1, 2, 2, 3]) == 2", "assert most_frequent([3, 1, 3, 1, 3]) == 3",
          "assert most_frequent([1, 2, 1, 2]) == 1", "assert most_frequent(['a', 'b', 'a']) == 'a'"],
         "most_frequent"),
    task("heldout_interleave", "heldout",
         'def interleave(a, b):\n    """Alternate elements of a and b. When one list is longer,\n    append its remaining elements at the end."""',
         ["assert interleave([1, 2], [3, 4]) == [1, 3, 2, 4]",
          "assert interleave([1], [2, 3, 4]) == [1, 2, 3, 4]",
          "assert interleave([], [1]) == [1]", "assert interleave([1, 2], []) == [1, 2]"],
         "interleave"),
    task("heldout_digit_sum", "heldout",
         'def digit_sum(n):\n    """Return the sum of the decimal digits of n (use absolute value)."""',
         ["assert digit_sum(123) == 6", "assert digit_sum(0) == 0",
          "assert digit_sum(-45) == 9", "assert digit_sum(999) == 27"], "digit_sum"),
    task("heldout_rotate_right", "heldout",
         'def rotate_right(lst, k):\n    """Rotate lst right by k positions. k may exceed len(lst).\n    Return a new list; empty list -> []."""',
         ["assert rotate_right([1, 2, 3, 4], 1) == [4, 1, 2, 3]",
          "assert rotate_right([1, 2, 3], 0) == [1, 2, 3]",
          "assert rotate_right([1, 2, 3], 3) == [1, 2, 3]",
          "assert rotate_right([1, 2, 3, 4, 5], 7) == [4, 5, 1, 2, 3]",
          "assert rotate_right([], 3) == []"], "rotate_right"),
]

# Reference solutions — for benchmark-correctness verification ONLY.
# Never ship these to a model or include them in training data.
ORACLES = {
    "dev_sum_of_squares": "    return sum(i * i for i in range(1, n + 1)) if n > 0 else 0",
    "dev_is_palindrome": ("    t = ''.join(c.lower() for c in s if c.isalnum())\n"
                          "    return t == t[::-1]"),
    "dev_word_count": "    return len(s.split())",
    "dev_fibonacci_list": ("    out = []\n    a, b = 0, 1\n    for _ in range(n):\n"
                           "        out.append(a)\n        a, b = b, a + b\n    return out"),
    "dev_merge_sorted": ("    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n"
                         "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n"
                         "        else:\n            out.append(b[j]); j += 1\n"
                         "    out.extend(a[i:]); out.extend(b[j:])\n    return out"),
    "dev_char_frequency": ("    d = {}\n    for c in s:\n        d[c] = d.get(c, 0) + 1\n    return d"),
    "dev_is_prime": ("    if n < 2:\n        return False\n    i = 2\n"
                     "    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True"),
    "dev_flatten_once": ("    out = []\n    for x in nested:\n"
                         "        out.extend(x) if isinstance(x, list) else out.append(x)\n    return out"),
    "dev_title_case": "    return ' '.join(w[:1].upper() + w[1:].lower() for w in s.split(' '))",
    "dev_gcd": ("    a, b = abs(a), abs(b)\n    while b:\n        a, b = b, a % b\n    return a"),
    "dev_unique_sorted": "    return sorted(set(lst))",
    "dev_caesar_shift": ("    out = []\n    for c in s:\n"
                         "        out.append(chr((ord(c) - 97 + k) % 26 + 97) if 'a' <= c <= 'z' else c)\n"
                         "    return ''.join(out)"),
    "heldout_run_length_encode": ("    out = []\n    i = 0\n    while i < len(s):\n        j = i\n"
                                  "        while j < len(s) and s[j] == s[i]:\n            j += 1\n"
                                  "        out.append(f'{s[i]}{j - i}')\n        i = j\n    return ''.join(out)"),
    "heldout_balanced_brackets": ("    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n"
                                  "    for c in s:\n        if c in '([{':\n            stack.append(c)\n"
                                  "        elif c in pairs:\n            if not stack or stack.pop() != pairs[c]:\n"
                                  "                return False\n    return not stack"),
    "heldout_second_largest": ("    vals = sorted(set(lst))\n"
                               "    return vals[-2] if len(vals) >= 2 else None"),
    "heldout_group_anagrams": ("    groups = {}\n    for w in words:\n"
                               "        groups.setdefault(''.join(sorted(w)), []).append(w)\n"
                               "    return sorted([sorted(g) for g in groups.values()])"),
    "heldout_pascal_row": ("    row = [1]\n    for _ in range(n):\n"
                           "        row = [1] + [row[i] + row[i + 1] for i in range(len(row) - 1)] + [1]\n"
                           "    return row"),
    "heldout_longest_common_prefix": ("    if not strs:\n        return ''\n    p = strs[0]\n"
                                      "    for w in strs[1:]:\n        i = 0\n"
                                      "        while i < len(p) and i < len(w) and p[i] == w[i]:\n"
                                      "            i += 1\n        p = p[:i]\n    return p"),
    "heldout_sieve": ("    if n <= 2:\n        return []\n    is_p = [True] * n\n    is_p[0] = is_p[1] = False\n"
                      "    for i in range(2, int(n ** 0.5) + 1):\n        if is_p[i]:\n"
                      "            is_p[i * i::i] = [False] * len(is_p[i * i::i])\n    return [i for i, p in enumerate(is_p) if p]"),
    "heldout_chunk_list": "    return [lst[i:i + k] for i in range(0, len(lst), k)]",
    "heldout_most_frequent": ("    best, best_n = None, -1\n    counts = {}\n"
                              "    for x in lst:\n        counts[x] = counts.get(x, 0) + 1\n"
                              "        if counts[x] > best_n:\n            best, best_n = x, counts[x]\n    return best"),
    "heldout_interleave": ("    out = []\n    for x, y in zip(a, b):\n        out += [x, y]\n"
                           "    out += a[len(b):] if len(a) > len(b) else b[len(a):]\n    return out"),
    "heldout_digit_sum": "    return sum(int(d) for d in str(abs(n)))",
    "heldout_rotate_right": ("    if not lst:\n        return []\n    k %= len(lst)\n"
                             "    return lst[-k:] + lst[:-k] if k else lst[:]"),
}


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build(out_path: Path) -> dict:
    assert len(TASKS) == 24, f"expected 24 tasks, got {len(TASKS)}"
    ids = [t["id"] for t in TASKS]
    assert len(set(ids)) == 24, "duplicate task ids"
    assert sum(1 for t in TASKS if t["split"] == "dev") == 12
    assert sum(1 for t in TASKS if t["split"] == "heldout") == 12

    # 1. Every oracle must pass every test (benchmark correctness, not a model result)
    for t in TASKS:
        r = run_candidate(t["prompt"], ORACLES[t["id"]], t["tests"], timeout=10)
        if not r.passed:
            raise SystemExit(f"ORACLE FAILED for {t['id']}: {r.error_class} {r.detail}")

    # 2. Every task must be solvable-in-principle check: prompt parses
    import ast as _ast
    for t in TASKS:
        _ast.parse(t["prompt"] + "\n    pass\n")

    bench = {
        "version": VERSION,
        "created": CREATED,
        "author": "FORGE project (Khushi) — original tasks",
        "license": "CC0-1.0",
        "frozen": True,
        "provenance": ("All 24 tasks and their tests were authored from scratch for FORGE on "
                       "2026-10-24. No task was copied from any public benchmark or dataset. "
                       "Reference solutions live only in build_heldout.py (quarantined, never "
                       "training data). Any edit to prompts or tests requires a version bump."),
        "splits": {"dev": "iteration allowed", "heldout": "final evaluation only; never tune on these"},
        "tasks": [],
    }
    for t in TASKS:
        bench["tasks"].append({
            **t,
            "prompt_sha256": sha256_hex(t["prompt"]),
            "tests_sha256": sha256_hex("\n".join(t["tests"])),
        })
    canon = json.dumps(bench, sort_keys=True, separators=(",", ":"))
    bench["benchmark_sha256"] = sha256_hex(canon)
    out_path.write_text(json.dumps(bench, indent=2) + "\n")
    return bench


if __name__ == "__main__":
    out = Path(__file__).parent / "forge_heldout_v1.json"
    bench = build(out)
    print(f"wrote {out}: {len(bench['tasks'])} tasks, benchmark_sha256={bench['benchmark_sha256'][:16]}…")
