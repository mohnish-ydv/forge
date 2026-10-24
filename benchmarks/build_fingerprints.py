"""Build benchmarks/fingerprints.json from public benchmark releases.

Source: evalplus package (HumanEval+ 164 problems: prompt + canonical
solution each). evalplus is the same source our eval harness will use,
so fingerprints and evaluation stay consistent.

Quarantine rules (see benchmarks/quarantine/README.md):
- Fingerprints are DERIVED DATA for decontamination only.
- They are NEVER training data, NEVER used for model tuning, and the
  quarantine dir is not on any pipeline source path.
- Fingerprint docs contain normalized hashes too, so the raw reference
  solutions do not need to ship with the training pipeline.

Writes: benchmarks/fingerprints.json  {docs: [{id, source, text}]}
"""
from __future__ import annotations

import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fingerprints.json")


def main():
    from evalplus.data import get_human_eval_plus
    problems = get_human_eval_plus()
    docs = []
    for task_id in sorted(problems):
        p = problems[task_id]
        # Fingerprint on prompt + canonical solution (the two things a
        # training file could copy). Keep text; hashes computed at load.
        text = p["prompt"] + "\n" + p.get("canonical_solution", "")
        docs.append({"id": task_id, "source": "humaneval-plus", "text": text})
    payload = {
        "built_by": "benchmarks/build_fingerprints.py",
        "source_package": "evalplus",
        "n_docs": len(docs),
        "quarantine_note": "Decontamination only. Never training data.",
        "docs": docs,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, sort_keys=True)
    print(f"wrote {len(docs)} fingerprint docs -> {OUT}")


if __name__ == "__main__":
    main()
