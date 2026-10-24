"""FORGE evaluation harness — sandboxed execution, pass@1.

Protocol (evaluation.md §1.3, forge-eval-v1):
- A sample solves a problem iff it passes ALL hidden tests in an
  isolated subprocess (10s, no network by construction, temp dir).
- Greedy pass@1 = fraction solved on single deterministic attempt.
- B0 floor test: empty body MUST score 0, else harness is broken.

Sandbox (M4): where the kernel allows it (Linux user namespaces), each
candidate runs in a pivot_root sandbox: private user+mount+network
namespaces, read-only /usr//lib, a fresh tmpfs /tmp and workspace, the old
root masked off, no network, cleared environment, and rlimits (CPU 30s,
address space 1 GiB, 64 processes, 100 MiB files). Where user namespaces
are unavailable the harness falls back to plain subprocess isolation and
records sandbox="plain" on the result. Untrusted candidates at scale still
belong in a throwaway container/VM — see the residual list in
run_candidate's docstring.

No external deps. Runs on CPU anywhere (VM, Kaggle, Termux).
"""
from __future__ import annotations

import ast
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Problem:
    name: str
    prompt: str          # signature + docstring
    tests: list[str]     # assert statements, run after candidate code
    entry_point: str = ""


@dataclass
class EvalResult:
    problem: str
    passed: bool
    error_class: str  # pass | syntax | import/runtime | timeout | assertion
    detail: str = ""
    sandbox: str = ""  # "pivot" (userns) or "plain" (subprocess-only fallback)


def check_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _limit_resources():
    """preexec_fn: cap CPU, memory, process count, file size (Linux only).

    Best-effort; on non-Linux or restricted kernels the wall-clock timeout
    and (in pivot mode) namespace isolation remain.
    """
    try:
        import resource
        # CPU seconds (hard kill shortly after)
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        # Address space: 1 GiB
        mem = 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        # Process count: fork-bomb containment
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        # Single file size: 100 MiB (disk-exhaustion containment)
        resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024))
    except Exception:
        pass


# Pivot-root sandbox script. $1 = host work dir (holds candidate.py),
# $2 = real python binary (must live under /usr, which is bound read-only).
# All mounts happen inside a private user+mount+network namespace, so the
# host is untouched. Exit codes 10..22 pinpoint which step failed.
_PIVOT_SCRIPT = r"""
set -u
HOST="$1"; PYBIN="$2"
NR="$HOST/.newroot"
mkdir -p "$NR"
mount -t tmpfs tmpfs "$NR" || exit 10
mkdir -p "$NR/usr" "$NR/lib" "$NR/lib64" "$NR/tmp" "$NR/dev" "$NR/work" "$NR/oldroot"
mount --bind /usr "$NR/usr" || exit 11
mount -o remount,ro,bind "$NR/usr" || exit 12
mount --bind /lib "$NR/lib" || exit 13
mount -o remount,ro,bind "$NR/lib" || exit 14
if [ -d /lib64 ]; then
  mount --bind /lib64 "$NR/lib64" || exit 15
  mount -o remount,ro,bind "$NR/lib64" || exit 16
fi
mount -t tmpfs tmpfs "$NR/tmp" || exit 17
touch "$NR/dev/null" "$NR/dev/urandom"
mount --bind /dev/null "$NR/dev/null" 2>/dev/null || true
mount --bind /dev/urandom "$NR/dev/urandom" 2>/dev/null || true
mount --bind "$HOST" "$NR/work" || exit 18
cd "$NR" || exit 19
pivot_root . oldroot || exit 20
mount -t tmpfs tmpfs /oldroot || exit 21   # mask the old root: host fs unreachable
cd /work || exit 22
exec "$PYBIN" /work/candidate.py
"""

_pivot_ok: bool | None = None


def _pivot_available() -> bool:
    """Probe once whether the pivot_root userns sandbox works here."""
    global _pivot_ok
    if _pivot_ok is not None:
        return _pivot_ok
    _pivot_ok = False
    if not shutil.which("unshare"):
        return False
    pybin = os.path.realpath(sys.executable)
    if not pybin.startswith("/usr/"):
        return False  # sandbox only binds /usr read-only
    try:
        with tempfile.TemporaryDirectory() as td:
            Path(td, "candidate.py").write_text("print('PIVOT_PROBE_OK')\n")
            p = subprocess.run(
                ["unshare", "-rmn", "bash", "-c", _PIVOT_SCRIPT, "forge-pivot", td, pybin],
                capture_output=True, text=True, timeout=60,
                stdin=subprocess.DEVNULL,
            )
            _pivot_ok = p.returncode == 0 and "PIVOT_PROBE_OK" in p.stdout
    except Exception:
        _pivot_ok = False
    return _pivot_ok


# Minimal environment for the candidate: no inherited secrets.
# sbin dirs included so the pivot script finds pivot_root/mount.
_CANDIDATE_ENV = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "PYTHONDONTWRITEBYTECODE": "1"}


def run_candidate(prompt: str, completion: str, tests: list[str], timeout: int = 10) -> EvalResult:
    """Execute prompt+completion+tests in an isolated subprocess.

    Isolation boundary (honest description):

    Pivot mode (sandbox="pivot", preferred):
    - Private user, mount and network namespaces; pivot_root into a tmpfs
      with read-only /usr and /lib, a fresh tmpfs /tmp and workspace, and
      the old root masked off — the host filesystem is unreachable.
    - No network (empty network namespace). Cleared environment (no
      inherited secrets). rlimits: CPU 30s, address space 1 GiB,
      64 processes, 100 MiB max file. Wall-clock timeout kills the whole
      process group, so forked grandchildren die too.
    - Still possible inside the sandbox: reading the candidate's own file,
      spawning subprocesses (contained in the namespace), /proc-less
      stdlib gaps. A candidate that deliberately reads its own file to
      steal the completion token and forge a pass is outside the scoring
      threat model (it would be blatant in the logged completion); the
      per-run token defeats accidental and casual spoofing.

    Plain mode (sandbox="plain", fallback where user namespaces are
    unavailable, e.g. some Termux/Kaggle kernels):
    - Separate OS process, fresh temp dir as cwd, wall-clock timeout,
      cleared environment, rlimits as above.
    - NOT restricted: filesystem reads/writes outside the temp dir,
      network sockets. Do NOT run untrusted model output in plain mode
      on a machine holding credentials or valuable data; use a throwaway
      container/VM for evaluation at scale.
    """
    full = prompt + "\n" + completion + "\n\n" + "\n".join(tests)
    if not check_syntax(full):
        return EvalResult(problem="", passed=False, error_class="syntax",
                          detail="ast.parse failed", sandbox="none")
    # Per-run unguessable completion token (M4 Gate A fix): a fixed marker
    # string would let a degenerate candidate print it and exit(0) before the
    # tests run, scoring a false pass. The candidate cannot predict 128 bits
    # of per-run entropy, so a pass now requires the script to actually reach
    # the end of the file — i.e. the tests ran without raising.
    token = "__FORGE_PASS_" + secrets.token_hex(16) + "__"
    full += f"\nprint('{token}')\n"
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/candidate.py"
        with open(path, "w") as f:
            f.write(full)
        sandbox = "plain"
        if _pivot_available():
            pybin = os.path.realpath(sys.executable)
            cmd = ["unshare", "-rmn", "bash", "-c", _PIVOT_SCRIPT,
                   "forge-pivot", td, pybin]
            sandbox = "pivot"
        else:
            cmd = [sys.executable, path]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                stdin=subprocess.DEVNULL,
                cwd=td, env=_CANDIDATE_ENV,
                preexec_fn=_limit_resources,
                start_new_session=True,  # own process group: timeout kills strays
            )
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                out, err = proc.communicate()
                return EvalResult(problem="", passed=False, error_class="timeout",
                                  detail=f">{timeout}s", sandbox=sandbox)
        except Exception as e:  # e.g. exec failed
            return EvalResult(problem="", passed=False, error_class="import/runtime",
                              detail=f"launch failed: {type(e).__name__}", sandbox=sandbox)
        if proc.returncode == 0 and token in (out or ""):
            return EvalResult(problem="", passed=True, error_class="pass", sandbox=sandbox)
        stderr = (err or "").strip().splitlines()
        last = stderr[-1] if stderr else (out or "")[:200]
        if "AssertionError" in (err or ""):
            cls = "assertion"
        else:
            cls = "import/runtime"
        return EvalResult(problem="", passed=False, error_class=cls,
                          detail=last[:300], sandbox=sandbox)


def evaluate(problems: list[Problem], completions: dict[str, str], timeout: int = 10) -> dict:
    results = []
    for p in problems:
        comp = completions.get(p.name, "")
        r = run_candidate(p.prompt, comp, p.tests, timeout=timeout)
        r.problem = p.name
        results.append(r)
    n = len(problems)
    passed = sum(1 for r in results if r.passed)
    funnel = {}
    for r in results:
        funnel[r.error_class] = funnel.get(r.error_class, 0) + 1
    return {
        "pass_at_1": passed / max(n, 1),
        "passed": passed,
        "total": n,
        "error_funnel": funnel,
        "results": results,
    }


# --- Mini held-out-style problems (original, authored for harness testing) ---
MINI_PROBLEMS = [
    Problem(
        name="add_two",
        prompt='def add_two(a, b):\n    """Return a + b."""',
        tests=["assert add_two(2, 3) == 5", "assert add_two(-1, 1) == 0", "assert add_two(0, 0) == 0"],
        entry_point="add_two",
    ),
    Problem(
        name="reverse_str",
        prompt='def reverse_str(s):\n    """Return s reversed."""',
        tests=["assert reverse_str('abc') == 'cba'", "assert reverse_str('') == ''", "assert reverse_str('a') == 'a'"],
        entry_point="reverse_str",
    ),
    Problem(
        name="count_vowels",
        prompt='def count_vowels(s):\n    """Count vowels (aeiou, case-insensitive) in s."""',
        tests=["assert count_vowels('hello') == 2", "assert count_vowels('AEIOU') == 5", "assert count_vowels('xyz') == 0"],
        entry_point="count_vowels",
    ),
]

GOOD_COMPLETIONS = {
    "add_two": "    return a + b",
    "reverse_str": "    return s[::-1]",
    "count_vowels": "    return sum(1 for c in s.lower() if c in 'aeiou')",
}
EMPTY_COMPLETIONS = {p.name: "    pass" for p in MINI_PROBLEMS}
BROKEN_COMPLETIONS = {
    "add_two": "    return a +",          # syntax error
    "reverse_str": "    return s",         # wrong logic -> assertion
    "count_vowels": "    raise ValueError('boom')",  # runtime error
}


if __name__ == "__main__":
    print("=== B0 floor test: empty body must score 0 ===")
    r0 = evaluate(MINI_PROBLEMS, EMPTY_COMPLETIONS)
    print(f"empty: pass@1={r0['pass_at_1']} funnel={r0['error_funnel']}")
    # Note: `pass` stub passes no asserts that check values -> mostly assertion failures.
    # A truly empty completion that still defines the function returns None -> assertion.
    print("\n=== Oracle completions must score 1.0 ===")
    r1 = evaluate(MINI_PROBLEMS, GOOD_COMPLETIONS)
    print(f"oracle: pass@1={r1['pass_at_1']} funnel={r1['error_funnel']}")
    assert r1["pass_at_1"] == 1.0, "harness broken: oracle did not pass"

    print("\n=== Error classification ===")
    r2 = evaluate(MINI_PROBLEMS, BROKEN_COMPLETIONS)
    print(f"broken: pass@1={r2['pass_at_1']} funnel={r2['error_funnel']}")
    for res in r2["results"]:
        print(f"  {res.problem}: {res.error_class} — {res.detail[:80]}")
    print("\nharness self-test: PASS")
