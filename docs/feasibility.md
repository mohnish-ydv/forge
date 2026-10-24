# FORGE — Feasibility Assessment, Comparison & Plan

**Date:** 2026-10-24 · **Author:** Khushi (lead research) · **Status:** Pre-training design, frozen for v0
**Sources:** `research/architectures.md`, `research/datasets.md`, `research/evaluation.md` — all claims below trace to those briefs. No FORGE benchmark scores exist yet; none are claimed.

---

## 1. Research-backed feasibility assessment

### Environment audit (verified in this VM, 2026-10-24)

| Capability | Verified? | Evidence |
|---|---|---|
| Create/modify files | Yes | `~/workspace/forge/` created, 150 GB disk, 149 GB free |
| Terminal / code execution | Yes | Python 3.12.3, git 2.43.0, 7.7 GB RAM in this VM |
| GitHub access | Yes | `github.com` 200 OK, API rate limit reachable |
| Kaggle / Colab reachability | Yes (network) | `kaggle.com` reachable; GPU quota is per-account, not testable from VM — see §1.2 |
| Browser research | Yes | Three briefs built from primary papers/repos |
| PyTorch installed | No | Not in base VM — install CPU wheel in venv for local smoke tests; GPU training happens on Kaggle |
| Phone/Termux | Assumed by user | Not directly testable from VM; deployment target is llama.cpp GGUF which is Termux-buildable |

**Conclusion:** Development, data-pipeline prototyping, tokenizer training, and CPU smoke tests are feasible *here*. Real pretraining is feasible *only* via free remote GPUs (Kaggle primary). No step requires a paid API.

### 1.2 Free compute — verified limits (from datasets brief, cross-checked 2026-10-24)

- **Kaggle:** ~30 GPU hrs/week (P100 or T4×2, 16 GB), ~20 TPU hrs/week, sessions ≤12 h, ~20 GB persistent disk. Quota is wall-clock, counts idle time. Phone verification required.
- **Colab free:** No published quota; typically T4 16 GB, ≤12 h/session, pre-emptible, not guaranteed. Treat as opportunistic overflow, never as the schedule.
- **HF Hub:** Free hosting/streaming for datasets/checkpoints. ZeroGPU (5 min/day free) is *not* training compute.
- **Termux/Android:** No GPU training. Use for coding, sample-scale cleaning, tokenizer experiments, and orchestrating Kaggle.

### 1.3 Is a useful small coder feasible on ₹0?

**Yes, with narrowed scope — and only with narrowed scope.**

Established evidence:
- Phi-1 (1.3B) reached a *reported* 50.6% HumanEval / 55.5% MBPP on only 7B tokens of textbook-quality data + exercise fine-tuning, with a conventional architecture. Its 350M sibling reached a reported 45% HumanEval in the same pipeline. Lesson: data quality and SFT dominate at this scale.
- Qwen2.5-Coder base→instruct jump at 0.5B is 28.0 → 61.6 HumanEval (+33.6 pts) from executor-filtered synthetic SFT alone. Pretraining scale alone does not explain capability.
- No credible HumanEval result was found for <100M from-scratch coders. TinyStories' <10M success does *not* transfer — stories have no execution semantics; code does.

**Feasibility boundary:** FORGE can realistically target:
- A 125M model that learns Python syntax, simple functions, and passes a small fraction of HumanEval+ / MBPP+ (single digits to low teens is a credible v0 outcome, not a promise).
- A 350M model that is the first plausibly useful Python coder *if* trained on ~1–2.5B high-quality tokens and followed by execution-filtered SFT.
- Phone deployment is feasible: 125M Q4_K_M ≈ 75 MB, 350M ≈ 210 MB, both fit 4–8 GB phones via llama.cpp. KV cache at 2k context must be budgeted (≈50 MB with GQA for 350M vs ≈200 MB with MHA).

**Not feasible on ₹0:** Replicating Qwen/StarCoder token counts (3–5.5T tokens), training ≥1.5B from scratch, or claiming general multilingual coding ability from a <500M model. FORGE must win on Python-first scope, data quality, and system design — not scale.

---

## 2. Comparison of three approaches

|  | A. From-scratch compact model | B. Adapt small pretrained open-weight model | C. System: model + retrieval / execution feedback |
|---|---|---|---|
| **What it is** | Train FORGE-125M/350M decoder-only Transformer on Stack-Edu Python-first data | Fine-tune / distil from Qwen2.5-Coder-0.5B/1.5B (Apache 2.0 — verify exact checkpoint licence before use) | FORGE model (A or B) + retrieval of code snippets, execution sandbox, 1-step repair loop, decomposition |
| **Training cost** | Highest: 125M/2.5B tok ≈ 29 T4-hrs (~1–1.5 Kaggle weeks); 350M/7B Chinchilla ≈ 228 hrs (~7–8 weeks) — v1, not v0 | Lowest: SFT/LoRA on 75K instructions in a few GPU-hrs; no pretraining | Same as underlying model + engineering time; inference costs more tokens/time |
| **Data needed** | 0.5–2.5B tokens pretraining + SFT set | 10–100K SFT examples | Same as model + retrieval index (e.g. FAISS over Stack-Edu) |
| **Expected coding ability** | Lowest initially; grows with tokens + SFT. Honest v0: syntax + simple functions | Highest immediately (inherits 5.5T-token pretraining). Instruct 0.5B reported 61.6 HumanEval — but contamination: unknown | +Δ over model-only at equal token budget, *if* retrieval/feedback is measured as a delta. Execution feedback fixes syntax/runtime errors reliably; retrieval helps API-heavy tasks |
| **Licence / distribution** | Cleanest if data is permissive-filtered Stack-Edu. FORGE owns weights | Inherits base model licence + teacher-data terms. Must log provenance | Inherits model licence; retrieval corpus licence matters if index is shipped |
| **Phone fit** | Excellent (designed for GGUF) | Good (0.5B Q4 ≈ 300 MB) | Retrieval index adds storage; execution sandbox needs Termux Python — feasible but heavier |
| **Scientific value** | Highest — tests the core question: how much coding per parameter? | Low for the from-scratch question; high as a baseline | Tests system vs parametric trade-off — FORGE's second core question |
| **Risks** | Slow, may plateau low; needs many Kaggle weeks | Not "FORGE from scratch"; may be disqualified if goal is strictly own model | Complexity, latency, easy to fool yourself by tuning on test set via feedback loop |

**Decision:** Pursue **A as the research core**, **B in parallel as the pinned baseline and practical fallback**, and **C as measured deltas over A/B** — never as a substitute for reporting model-only scores. This matches the briefs: B gives an honest comparator from day one; C is evaluated strictly as `score(C) − score(model-only)` at equal token budget.

Baselines (from evaluation brief):
- **B0 floor:** empty body / most-common template / copy-retrieval — must score ~0 or harness is broken.
- **B1 control:** 30–50M from-scratch at equal tokens — any innovation must beat it.
- **B2 reference:** pinned open-weight 0.5–1.5B, labelled `contamination: unknown`.
- **B3 headline:** FORGE model-only greedy pass@1.
- **B4 system:** B3 + retrieval / feedback / decomposition, reported as delta only.

---

## 3. Proposed first architecture

**One codebase, two configs — Llama-style decoder-only Transformer.**

### FORGE-125M (first real experiment — ablations, fast iteration)
- Layers 18, d_model 768, heads 12 Q / 3 KV (GQA factor 4), SwiGLU intermediate 2048, vocab 16,384 (byte-level BPE, tied embeddings), context 2048, RoPE θ=10,000, pre-norm RMSNorm, no bias (except optional QKV bias ablation)
- **Verified by `src/model.py` arithmetic: 124.09M total (111.5M non-embedding). Q4_K_M ≈ 73 MB, KV cache @2k (FP16) ≈ 28 MB.**

### FORGE-350M (first usefulness test — after 125M pipeline is proven)
- Layers 28, d_model 1024, heads 16 Q / 4 KV (GQA factor 4), SwiGLU intermediate 2816, vocab 32,768 (tied), context 2048
- **Verified: 349.23M total (315.7M non-embedding). Q4_K_M ≈ 206 MB, KV cache @2k ≈ 59 MB.**
- Smoke config FORGE-30M (8 layers, d=512, vocab 8k): verified 28.32M, Q4 ≈ 17 MB — for pipeline tests only.

**Why this is the best initial experiment:**
1. Only architecture with mature free tooling end-to-end (PyTorch → HF → llama.cpp GGUF) from 10M to 1.5B.
2. Efficiency stack is converged and low-risk: RMSNorm, RoPE, SwiGLU, GQA, tied embeddings, BF16 + FlashAttention-2/SDPA. Each saves parameters, KV cache, or training time without novelty risk.
3. Tokenizer arithmetic forces small vocab: Qwen's 151k vocab would cost ~136M params in a 0.5B model (~27%) for multilingual coverage FORGE doesn't need. 16k/32k keeps embeddings at ~10% of budget.
4. GQA factor 4 cuts KV cache ~4× (350M at 2k ctx: ~201 MB MHA → ~50 MB GQA) — decisive on phones.
5. Alternatives deferred with reasons: Mamba/hybrid (lags on copying/in-context learning in controlled 8B study, thin tooling), MoE (storage = total params, not phone-small), encoder-decoder (splits budget, no generation advantage), sliding-window/MLA (solves >4k context problems FORGE doesn't have yet).

**Ablations queued (not defaults):** deep-and-thin MobileLLM shape, block-wise sharing, FIM at 50% PSM. Each needs a pre-registered threshold vs B1 control.

---

## 4. Data strategy

**Principle:** Spend CPU time on filtering so GPU time is spent on good tokens. Run cleaning on Kaggle CPU, not GPU quota.

### Sources (v0, Python-first)
- **Primary:** `HuggingFaceTB/stack-edu` Python slice, `int_score ≥ 3`. Stack-Edu is Stack v2 filtered by educational classifiers; filtering improved MultiPL-E Python 20.7 → 25.6 in SmolLM2 ablations. Per-language: Python 50.6B → 21.8B tokens after filtering — stream a subset, never download raw Stack v2 (67.5 TB, gated, SWHID-only).
- **Bootstrap:** Python-Edu (~4B tokens, SmolLM corpus component) — exactly FORGE scale.
- **Web mix (10–15%):** FineWeb-Edu sample-10BT (ODC-By 1.0) for docstring/instruction grounding. Raw FineWeb dilutes code per token.
- **SFT (after pretraining):** Magicoder-OSS-Instruct-75K (listed MIT; teacher was gpt-3.5 — log provenance, execution-filter before use) + self-generated, execution-filtered problems. Evol-Instruct-110K (Apache-2.0) second.

**Do NOT use at v0:** Raw Stack v2, unfiltered GitHub scrapes, Pile/RedPajama full, any unclear-licence or research-only set in the redistributed mix. Stack v3 blog claims are unverified — ignore until an official card exists.

### Pipeline (DataTrove + OSS tools, in order)
Licence filter (permissive-only, drop `no_license` for clean release) → language/file filter → heuristic quality → exact SHA256 dedup → MinHash+LSH near-dedup (5-gram shingles, Jaccard 0.7) → PII/secrets redaction (Presidio, StarPii, ScanCode) → decontamination → Edu scoring → tokenise/pack (2k blocks).

### Contamination — 3 layers (validity gate, not optional)
1. 10-gram + 13-gram normalised overlap vs HumanEval/MBPP/APPS/MultiPL-E prompts + solutions. Any hit → drop document, log benchmark ID.
2. MinHash vs benchmark solutions (Jaccard ≥0.7) + repo-name blocklist (`humaneval`, `mbpp`, `evalplus`, mirror names quarantined wholesale).
3. Embedding / logic check awareness: n-grams miss rephrased contamination (Phi-1 finding; Deng et al. F1≈0 on rephrased). Mitigation is FORGE-Heldout (120 private problems, hash-committed at freeze, never trained on) as the fair generalisation test.

Splits are **by repo, never by file**. Publish decontamination counts per release.

### Budgets (Chinchilla ~20 tok/param, T4 estimates with haircut)
- Smoke: 125M / 0.5B tok ≈ 6 T4-hrs
- **v0 target: 125M / 2.5B tok (Chinchilla) ≈ 29 T4-hrs ≈ 1–1.5 Kaggle weeks**, storage ≈ 10–12 GB (fits 20 GB persistent)
- v1 target: 350M / 1B tok probe first; full 7B Chinchilla (≈228 hrs) only if v0 justifies it. 7B tok must stream shards from HF Hub.

### Curriculum & synthetic data
3 phases (foundations → general → annealing with high-quality algorithmic/synthetic), shuffled *within* phase. Ablate curriculum vs shuffled — do not assume it helps.
Synthetic: OSS-Instruct-style generation from real snippets, filtered by **execution only** (pytest in isolated subprocess, dual test sets, FAISS dedup). No synthetic example enters SFT without passing its own tests.

---

## 5. Benchmark & experiment plan

### Frozen protocol (forge-eval-v1)
- **Headline:** Greedy pass@1 (n=1, T=0) on **EvalPlus** — HumanEval+ (164) + MBPP+ (378) = 542 problems. Report base *and* `+` scores; the delta is fragility. EvalPlus has ~80× (HumanEval+) / ~35× (MBPP+) more tests than base; base pass@k overstates correctness by 19.3–28.9% relative.
- **Diagnostic:** Sampled pass@1 / pass@10 (n=20, T=0.8, top-p 0.95, 3 seeds, mean ± sd).
- **Protected final:** FORGE-Heldout (120, ≥10 tests each, hash-frozen) + HumanEval+ re-run, once per major version, never for checkpoint selection.
- **Dev (iterable):** 90-problem MBPP val slice, greedy, every checkpoint.
- **Execution:** Sandboxed subprocess, 10 s, 512 MB, no network. Prompting frozen (HumanEval zero-shot completion; MBPP+ 3-shot fixed examples). Any change = new eval version.

pass@k estimator (Chen et al.): `E[1 − C(n−c,k)/C(n,k)]`, c = # passing of n samples.

### Always reported alongside correctness
Syntax success %, execution success %, partial test pass %, duplicate/template rate (MinHash >0.8), contamination flags, truncation rate; plus systems metrics: total/non-embedding params, checkpoint MB (FP16/Q8/Q4), peak RAM, latency, tok/s (device named), GPU-hrs, tokens seen, FLOPs ≈ 6ND.

### Uncertainty — do not fool yourself
HumanEval alone (n=164): 30% pass@1 has Wilson 95% CI ±7.0 pp. **Deltas <7 pp on HumanEval alone are noise.** Combined 542: ±3.9 pp at 30%. Every score ships with Wilson/bootstrap CI, decoding conditions, model hash, and error-class breakdown.

### Experiment discipline (per optimisation)
Record: observed limitation → mechanism → hypothesis → experiment + baseline → compute/data cost → success threshold (pre-registered) → failure modes → actual result → next decision. Smallest useful experiment first. Ablations vs B1 control at equal tokens.

---

## 6. Prioritised roadmap with acceptance criteria

| Milestone | Deliverables | Acceptance criteria |
|---|---|---|
| **M0 Environment** ✅ | This audit, repo scaffold | Files/terminal/GitHub verified; research briefs saved |
| **M1 Model + tokenizer code** | `src/model.py` (RMSNorm/RoPE/SwiGLU/GQA, tied), `src/tokenizer.py` (BBPE train), unit tests | Tests pass on CPU; param count matches config ±1%; tokenizer lossless on code samples incl. indent runs; fertility reported vs GPT-2/Qwen |
| **M2 Data pipeline (sample)** | Cleaning + decontamination on 10K-file sample, stats report | Pipeline runs on Kaggle CPU; dedup + decontamination counts logged; zero benchmark 10-gram hits in cleaned sample (by construction, verified) |
| **M3 Smoke training** | 30M model, 10M tokens, CPU/GPU short run, checkpoint + resume | Loss decreases; checkpoint saves full optimiser+RNG+dataloader state; kill-resume test passes (resume loss within tolerance) |
| **M4 Eval harness** | EvalPlus fork, FORGE-Heldout draft (hash-committed), B0 floor test | B0 empty-body scores 0; harness reproduces a published baseline score within CI on same checkpoint (or labels why not) |
| **M5 FORGE-125M v0 pretraining** | 125M on 0.5B tok (probe) → 2.5B tok (Chinchilla) on Kaggle, checkpoints every 30 min | Checkpoints pushed to HF Hub; training log with GPU-hrs/tokens; no single-session dependency |
| **M6 Baseline evaluation** | B1/B2/B3 scores on forge-eval-v1 with CIs | Full report with conditions + error breakdown; contamination flags included/excluded both reported |
| **M7 SFT + system deltas** | Execution-filtered SFT; B4 retrieval/feedback deltas | SFT data 100% execution-verified; B4 reported only as delta over B3 at equal token budget |
| **M8 Phone deployment** | GGUF Q4_K_M (+ Q8/Q5 A/B), llama.cpp on Termux test | Quant A/B: ship smallest quant with ≤1.5 pp pass@1 drop; peak RAM + tok/s measured on named device; ctx 1024 default |
| **M9 Reproducibility audit** | Configs, seeds, data hashes, final report | Fresh checkout reproduces eval of released checkpoint within CI; no paid dependency |

**Ordering rule:** No milestone's GPU spend starts until the previous milestone's acceptance gate passes. Do not overwrite the only known-good checkpoint (last-2 rotation, atomic rename).

---

## Appendix — What success means for FORGE

Success is *not* beating Qwen2.5-Coder-1.5B. Success is:
1. A reproducible pipeline that trains a 125M Python coder from scratch on free compute, with honest EvalPlus + Heldout scores and CIs.
2. A measured answer to: how much does data quality / SFT / system feedback each contribute per parameter?
3. A GGUF model that runs on a phone in Termux and solves simple programming problems single-attempt, with known failure modes.

If 125M plateaus near zero on Heldout after 2.5B quality tokens + verified SFT, that is a *result* — preserve the evidence, publish the negative, and pivot compute to Path B/C with the same harness.
