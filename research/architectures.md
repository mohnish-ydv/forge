# FORGE — Compact Code Models & Efficient Architectures: Research Brief

**Project:** FORGE — a small, genuinely capable coding model built under severe constraints
**Constraints assumed throughout:** ₹0 wherever technically possible; primary dev environment is Android / Termux; free remote GPUs only when actually accessible (Kaggle / Colab-class); no paid API may be a requirement for training or inference; final model must be compact, portable, and runnable on ordinary hardware including phones.
**Date of this brief:** 2026-10-24 (Asia/Calcutta)
**Status:** Literature brief only. No FORGE training run is claimed here. No benchmark in this document is a FORGE result.

> **How to read the evidence tags**
> - **[Established]** — supported by a peer-reviewed paper, a primary technical report, or convergence across multiple independent model families.
> - **[Reported]** — a number or claim reported by the model's own authors in their technical report / model card. Real, citable, but *not* independently audited and not necessarily comparable across papers (different harnesses, prompts, decoding).
> - **[Hypothesis — FORGE to test]** — plausible for FORGE, not established for this setting. Must be turned into an ablation with a pre-registered threshold before it is believed.
>
> **Auditor's warning on all benchmark tables below:** HumanEval / MBPP numbers differ by (a) base vs. instruction-tuned checkpoint, (b) greedy vs. sampled pass@1, (c) original vs. EvalPlus (HumanEval+ / MBPP+) test suites, (d) prompt format, and (e) decontamination. Comparing a base model in one paper with an instruct model in another is a category error. This brief keeps base and instruct rows in separate tables for that reason.

---

## Executive summary — the decision in one page

1. **Build a decoder-only Transformer in the modern Llama style. Do not start with Mamba, RWKV, or an encoder–decoder.** Every strong small code model with public weights, a technical report, and mature phone inference support — Qwen2.5-Coder, StarCoder2, DeepSeek-Coder, TinyLlama, Phi — is a decoder-only Transformer. The tooling FORGE can actually use for free (PyTorch, Hugging Face `transformers`, llama.cpp / GGUF) is built around that shape. Alternatives are interesting research, poor first bets on ₹0.
2. **Data quality dominates architecture novelty at this scale [Established].** Phi-1 reached a reported 50.6% HumanEval pass@1 at 1.3B not with a new architecture — its authors explicitly call architecture and training "fairly conventional" — but with 7B tokens of filtered "textbook-quality" data plus synthetic exercises. Phi-1-*base* before exercise fine-tuning and phi-1-*small* (350M) show the same lesson from the other side (see §2).
3. **The realistic FORGE ladder is 30M (pipeline smoke test) → ~125M (first real experiment) → ~350M (first plausibly useful coder) → 0.5–1.5B only if free compute actually materialises.** A 1.5B model is not "small" to *train*: Qwen2.5-Coder was trained on 5.5 trillion tokens. FORGE cannot replicate that token count on free tiers and should not pretend otherwise. FORGE's edge, if any, must come from narrow scope (e.g. Python-first), high-quality / execution-filtered data, and a system around the model (retrieval + execution feedback) — not from out-scaling Qwen.
4. **Tokenizer: train a byte-level BPE (BBPE) tokenizer on FORGE's own code-heavy corpus, vocab 16k for the ~125M model and 32k for the ~350M model, with tied embeddings.** Do *not* copy Qwen's 151,646-token vocabulary into a sub-billion from-scratch model — the embedding matrix alone would eat the parameter budget (arithmetic in §3.2). Do *not* use pure byte-level (256 tokens) either — sequences become ~3–4× longer and attention cost is quadratic in sequence length.
5. **Adopt the converged efficiency stack from day one:** pre-norm RMSNorm, RoPE, SwiGLU, Grouped-Query Attention (GQA), tied embeddings below ~1B, no bias in most linear layers (Qwen's QKV bias is the notable exception), FlashAttention-2 where the GPU supports it, and BF16 training / Q4_K_M GGUF for phone inference. Each item is justified in §4. Treat sliding-window attention, layer sharing, and FIM (fill-in-the-middle) as *second-phase ablations*, not defaults.

---

## 1. Architecture options ranked for FORGE constraints

Scoring is for FORGE specifically — ₹0, Termux/Android development, intermittent free GPUs, a small team, need for llama.cpp deployment — not for "best architecture in the abstract".

| Rank | Option | FORGE fit | Why | Main risk |
|---|---|---|---|---|
| **1** | **Decoder-only Transformer, Llama-style** (RMSNorm pre-norm + RoPE + SwiGLU + GQA) | **Build this** | Used by TinyLlama, Qwen2.5-Coder, StarCoder2, DeepSeek-Coder, Phi. Maximal free tooling: reference implementations in PyTorch / lit-GPT / nanoGPT lineage, HF support, llama.cpp GGUF export, largest pool of debugging knowledge. Scales cleanly from 10M to 1.5B with only config changes. | None structural. Risk is data / compute, not architecture. |
| **2** | **Same Transformer, "deep-and-thin" + sharing variant (MobileLLM-style)** | **Ablate against Rank 1 at 125M/350M** | MobileLLM shows *at sub-billion scale architecture choices matter more than folklore suggests*: deep-and-thin (e.g. 30 layers for a 125M model), embedding sharing, and GQA gave a reported +2.7% / +4.3% accuracy over prior 125M / 350M state of the art on their commonsense suite; immediate block-wise weight sharing (MobileLLM-LS) added a further reported +0.7% / +0.8% with no size increase and only marginal latency [Reported]. | Deeper = slower step time on a small GPU and harder optimisation. Sharing can cap capacity. Must be *measured* on code tasks — MobileLLM's headline numbers are not code benchmarks. |
| **3** | **Adapt a small pretrained open-weight coder** (e.g. Qwen2.5-Coder-0.5B / 1.5B, Apache 2.0) as (a) teacher / data filter, (b) distillation source, (c) Path-B baseline | **Do this in parallel as a baseline, not as "FORGE the model"** | Immediately gives FORGE a strong, honestly-measured comparator and a practical coding system while the from-scratch model is weak. Qwen2.5-Coder 0.5B/1.5B are Apache 2.0 per their technical report table; always re-verify the licence of the exact checkpoint before redistribution. | This is not training FORGE from scratch. Storage / download on a phone and Kaggle disk limits are real. Distillation data generated by a teacher inherits the teacher's licence and contamination profile — log provenance. |
| **4** | **Hybrid Transformer + SSM (Mamba-2) / pure Mamba** (e.g. Codestral Mamba 7B as existence proof) | **Defer — revisit after a working Transformer baseline** | SSMs have linear (not quadratic) complexity in sequence length and no KV cache; long-context characterisation work finds Mamba-2 overtakes a Transformer only at long context (~32k in that study), while the Transformer was faster at short context [Reported]. A large controlled comparison (8B, same data, up to 3.5T tokens) found pure SSMs match Transformers on many tasks but *lag on copying, in-context learning, and long-context reasoning* — all load-bearing for code — while a hybrid (43% Mamba-2 / 7% attention / 50% MLP) beat the pure Transformer on average in that study [Reported]. | Training kernels, HF / llama.cpp support, and debugging experience are far thinner than for Transformers. Codestral Mamba is 7B — far above FORGE's phone / free-GPU training envelope. A failed Mamba experiment on FORGE's budget teaches less than a failed Transformer experiment because fewer people can help debug it. |
| **5** | **Encoder–decoder (T5 / CodeT5-style), RNN / RWKV, retrieval-only (no parametric model)** | **Do not build as the main model** | Encoder–decoders are defensible for infilling / translation-style tasks but have no advantage for autoregressive code generation and split the parameter budget between encoder and decoder. Retrieval-only is a *system* baseline FORGE should measure (Path C in the project plan), not a model. | Wastes the scarcest resource (experiment cycles) on the least-supported path. |

**Sources for the ranking:**
- TinyLlama is explicitly "based on the decoder-only Transformer" with Llama-2 architecture: https://arxiv.org/html/2401.02385v2
- Qwen2.5-Coder architecture table (all sizes share the same Transformer family, differing in hidden size / layers / heads): https://ar5iv.labs.arxiv.org/html/2409.12186
- StarCoder2 (Grouped Query Attention, sliding window, FIM): https://arxiv.org/html/2402.19173v1
- DeepSeek-Coder (decoder-only, RoPE, GQA in large models, FlashAttention-2, FIM): https://arxiv.org/html/2401.14196v1
- MobileLLM (deep-and-thin, embedding sharing, GQA, block-wise sharing): https://proceedings.mlr.press/v235/liu24ce.html
- Mamba vs. Transformer long-context characterisation: https://arxiv.org/html/2507.12442v4
- TinyStories as a *methodological* precedent (see box below): https://arxiv.org/pdf/2305.07759

> **What TinyStories does and does not prove for FORGE [Established, with limits]**
> TinyStories showed models *below 10M parameters*, even a 1-layer model, can produce fluent, consistent multi-paragraph stories — *when the domain is radically narrowed* (vocabulary and world of a 3–4-year-old, synthetic data generated by GPT-3.5/4). The transferable lesson for FORGE is **scope narrowing + synthetic, curriculum-shaped data can make a tiny model look capable inside that scope**. The non-transferable part: English stories have no execution semantics. Code either passes its tests or it does not, identifiers and APIs form a far larger effective vocabulary, and a model that is "fluent" Python but wrong is worthless. FORGE may copy TinyStories' *experiment style* (tiny models trainable in <1 day on one GPU, systematic width/depth ablations) without expecting TinyStories' parameter counts to suffice for code.

### 1.1 What about MoE, MLA, and other fashionable variants?

- **Mixture-of-Experts:** total parameters ≠ active parameters. On a phone, *total* parameters determine storage and RAM residency (all experts must be stored, even if sparsely activated). MoE helps cloud serving economics; it does not make a model phone-small. Skip for FORGE v1. **[Established reasoning — arithmetic, not a benchmark]**
- **Multi-head Latent Attention (MLA, DeepSeek-V2 style):** compresses the KV cache via a low-rank latent. Genuinely promising, but adds implementation complexity and is validated mainly at large scale. GQA achieves most of the KV saving FORGE needs at 2k context with a 20-line change. Revisit only if FORGE moves to ≥8k context. **[Hypothesis — FORGE to test, later]**
- **Sliding-window attention (SWA):** StarCoder2-3B uses a 16,384-token context with a 4,096-token sliding window [Reported]. SWA caps KV-cache growth, but code has exact long-range dependencies (a definition 3,000 tokens earlier matters verbatim). At FORGE's planned 1k–2k context, full attention over the window is affordable — SWA solves a problem FORGE does not yet have. Ablate it when context is extended.

---

## 2. Model sizes and documented coding performance

### 2.1 Base models — HumanEval / MBPP (pass@1, as reported by each paper)

All rows are **base (pretrained, not instruction-tuned)** unless marked. Do not compare across rows without reading the harness note under the table.

| Model | Params | HumanEval | HumanEval+ | MBPP | Source |
|---|---:|---:|---:|---:|---|
| Qwen2.5-Coder-0.5B (base) | 0.5B | 28.0 | 23.8 | —* | Qwen2.5-Coder Tech. Report, Table in §7 |
| Qwen2.5-Coder-1.5B (base) | 1.5B | 43.9 | 36.6 | —* | Same |
| Qwen2.5-Coder-3B (base) | 3B | 52.4 | 42.7 | —* | Same |
| StarCoder2-3B (base) | 3B | 31.7 | 27.4 | 57.4 (MBPP+ 47.4) | StarCoder2 paper, Table 9 (greedy) |
| DeepSeek-Coder-1.3B (base) | 1.3B | see note † | — | — | DeepSeek-Coder paper |
| phi-1 (1.3B, after exercise fine-tuning) | 1.3B | **50.6** | n/a | **55.5** | Phi-1 paper abstract |
| phi-1-small (same pipeline, smaller) | 350M | **45** (as stated in abstract for the fine-tuned small model) | n/a | n/a | Phi-1 paper abstract |
| Phi-3-mini | 3.8B | 58.5 [Reported, via secondary summary of Microsoft's report] | n/a | n/a | See sources |

\* The Qwen report's base-model table as extracted contains MBPP, MBPP+, and 3-shot MBPP columns whose alignment is ambiguous in plain-text extraction; this brief therefore does not quote a Qwen base MBPP number rather than quote a possibly mis-aligned one. The HumanEval / HumanEval+ columns are unambiguous. Read the PDF table directly before using any Qwen MBPP figure in a FORGE decision: https://arxiv.org/pdf/2409.12186

† DeepSeek-Coder-1.3B illustrates the harness problem perfectly: it appears with *different* HumanEval numbers in the Qwen2.5-Coder comparison table and in the StarCoder2 comparison table, because each paper re-evaluated (or cited) it under its own settings. This brief does not pick one as "the" score. FORGE's lesson: **pick one harness (recommendation: EvalPlus), pin its version, and re-evaluate every baseline FORGE compares against in that harness.**

**Sources:**
- Qwen2.5-Coder base + instruct tables: https://arxiv.org/html/2409.12186v2 and PDF https://arxiv.org/pdf/2409.12186
- StarCoder2 Table 9 (greedy pass@1, base models only): https://arxiv.org/html/2402.19173v1
- Phi-1 (50.6% HumanEval, 55.5% MBPP, 7B training tokens, 1.3B params; 350M small model 45% HumanEval): https://arxiv.org/pdf/2306.11644 and HTML https://ar5iv.labs.arxiv.org/html/2306.11644
- DeepSeek-Coder (1.3B–33B, 2T tokens, 87 languages, FIM, 16k window): https://arxiv.org/pdf/2401.14196
- Phi-3-mini 58.5% HumanEval as reproduced from Microsoft's Phi-3 report (treat as developer-reported, not independently verified in this brief): https://systems-analysis.info/eng/Phi_%28Microsoft%29 — note: this is a secondary summary; verify against Microsoft's own Phi-3 technical report before citing it in a publication.

### 2.2 Instruction-tuned models — why the jump matters more than the architecture

| Model (instruct) | Params | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-0.5B-Instruct | 0.5B | 61.6 | 57.3 | 52.4 | 43.7 |
| Qwen2.5-Coder-1.5B-Instruct | 1.5B | 70.7 | 66.5 | 69.2 | 59.4 |
| Qwen2.5-Coder-3B-Instruct | 3B | 84.1 | 80.5 | 73.6 | 62.4 |

Source: Qwen2.5-Coder Technical Report, instruct-model table (EvalPlus): https://arxiv.org/html/2409.12186v2

**Reading [Established pattern, not a single-paper artefact]:** the *same* 0.5B weights go from 28.0 (base) to 61.6 (instruct) HumanEval. That +33.6-point gap is not pretraining scale — it is supervised fine-tuning on synthetic, executor-filtered instruction data (the Qwen report describes fine-tuning with synthetic data from its predecessor with an executor retaining only executable code). Phi-1 shows the same shape: textbook pretraining alone was not the headline; exercise fine-tuning produced the headline and, per its authors, "unlocked" further capabilities. **FORGE implication:** budget experiment cycles for the SFT / execution-filtered stage at least as carefully as for pretraining. A mediocre base + excellent verified SFT data will beat an excellent base + sloppy SFT data on HumanEval-style tests — and the reverse risk (SFT data contaminating the test set) must be controlled by decontamination, §6 of the project plan.

### 2.3 What each scale can and cannot do — honest synthesis

These are *generalisations from the tables above*, stated with their limits. None is a promise about FORGE's model, which will be trained on orders of magnitude fewer tokens than any model in the tables.

- **~10–30M — [Established for narrow synthetic domains; Hypothesis for code]** Enough for TinyStories-style fluency in a tiny vocabulary. For code: enough to learn Python *syntax* (balanced brackets, indentation, keyword placement) and memorised idioms, not enough — on any published evidence found for this brief — for reliable functional correctness on unseen problems. **Use in FORGE as:** pipeline smoke tests, tokenizer / training-loop debugging, scaling-law probes. *No source found in this research reports a credible HumanEval pass@1 for a <100M code model; if FORGE finds one, treat it as suspect until its harness and decontamination are inspected.*
- **~125M — [Hypothesis — FORGE to test]** The classic GPT-Neo / CodeGen-350M era showed small models of this class struggle to produce coherent long outputs when trained on broad data (the motivating observation of TinyStories). With Python-only, textbook-style data, a 125M FORGE model plausibly learns syntax + short templated functions. Expect HumanEval pass@1 in the *single digits to low teens at best* without heavy SFT — this is an extrapolation, not a cited score. **Use in FORGE as:** the first *real* experiment — cheap enough to ablate tokenizer, depth/width, and data mix.
- **~350M — [Reported anchor: phi-1-small, 45% HumanEval]** The most encouraging data point for FORGE: a 350M model trained with phi-1's *data pipeline* (not FORGE's yet-unknown data) reportedly reached 45% HumanEval after fine-tuning. That required GPT-3.5-generated textbooks — a teacher FORGE may not use as a *requirement* (₹0 / no paid API rule), though permissively-licensed already-released synthetic datasets may be usable after licence checks. **Use in FORGE as:** the first size at which "genuinely useful for simple functions" is a testable claim rather than a hope.
- **0.5B — [Reported anchor: Qwen2.5-Coder]** Base 28.0 / instruct 61.6 HumanEval. The instruct number required 5.5T pretraining tokens + industrial SFT. FORGE at 0.5B with, say, 5–20B tokens should expect to land *far* below both numbers initially. The value of 0.5B for FORGE is as a *fine-tuning / distillation target* (Path B) and as the largest size that is still comfortable on a mid-range phone at Q4 (see §2.4).
- **1.1–1.5B — [Reported anchors: TinyLlama 1.1B (general), phi-1 1.3B (code, 50.6), Qwen2.5-Coder-1.5B (base 43.9 / instruct 70.7)]** This is where published small coders become genuinely capable on HumanEval-style tasks — *given* industrial token counts (TinyLlama: 3T tokens for a 1.1B general model; Qwen2.5-Coder: 5.5T). TinyLlama's own accounting — ~90 days on 16× A100-40G for its full run — is the clearest public reminder that "small model" ≠ "cheap to pretrain from scratch". **Use in FORGE as:** an aspirational ceiling and a fine-tuning target, not a from-scratch starting point.
- **3B+ — out of scope for from-scratch FORGE training; in scope as a teacher / oracle on free cloud GPUs when available.** StarCoder2-3B base (31.7) being *below* Qwen2.5-Coder-3B base (52.4) in their respective reports also shows generation / data-recipe effects swamping size effects within a factor of ~2.

### 2.4 Phone deployment arithmetic [Established — arithmetic + reported device measurements]

Weight storage is `params × bytes-per-param`. Quantisation changes bytes-per-param; it does not change parameter count.

| Params | FP16 (2 B) | INT8 (1 B) | Q4_K_M (~0.55–0.6 B effective, incl. scales) | Practical phone verdict |
|---:|---:|---:|---:|---|
| 125M | 250 MB | 125 MB | ~70–80 MB | Trivial. Runs anywhere. |
| 350M | 700 MB | 350 MB | ~200 MB | Comfortable on 4 GB-RAM phones. |
| 0.5B | 1.0 GB | 500 MB | ~300 MB | Comfortable. Sweet spot with KV cache + app overhead. |
| 1.1B | 2.2 GB | 1.1 GB | ~650 MB | Feasible on 6–8 GB phones via llama.cpp `mmap`; watch thermals. |
| 1.5B | 3.0 GB | 1.5 GB | ~0.9–1.2 GB | Feasible on flagship / upper-mid phones; a secondary source reports ~1.2 GB VRAM for Q4 Qwen2.5-Coder-1.5B: https://localaimaster.com/models/qwen-2-5-coder-1-5b |
| 3B | 6.0 GB | 3.0 GB | ~1.8 GB file, ~2.1 GB peak RAM, ~12–15 tok/s on Pixel 8 [Reported, single blog benchmark] | Possible, not comfortable. Thermal throttling is the binding constraint, not file size. |

Device-measurement sources (treat speeds as device- and build-specific, not universal):
- 3B Q4_K_M sizing / Pixel 8 vs. iPhone 15 Pro speeds, `mmap` guidance, thermal notes: https://dev.to/software_mvp-factory/on-device-llm-inference-via-kmp-and-llamacpp-4pec
- llama.cpp on Android (Q4 essential, Big.LITTLE thread tuning, Vulkan variability, background-kill risk): https://deepwiki.com/max-krasnyansky/llama.cpp/8.1-android
- ExecuTorch Llama 3.2 1B on OnePlus 12 (project README table: 1B BF16 decode 19.2 tok/s, RSS 3,185 MiB; SpinQuant build 50.2 tok/s, RSS 1,921 MiB — quantisation + kernels matter as much as parameter count): https://android.googlesource.com/platform/external/executorch/+/HEAD/examples/models/llama/README.md

**KV cache can rival weight size at long context — do the arithmetic for FORGE's config, not a generic model's:**
`KV bytes = 2 (K and V) × layers × KV-heads × head-dim × bytes × context-tokens`.
Worked example for a plausible FORGE-350M (24 layers, head-dim 64, BF16, 2,048 context):
- GQA with 4 KV heads: 2 × 24 × 4 × 64 × 2 × 2048 ≈ **50 MB** — negligible.
- Full MHA with 16 KV heads: ≈ **201 MB** — still survivable, but 4× larger for no established quality gain at this scale.
At 512-token context (enough for single-function generation) both numbers divide by 4. **Conclusion: at FORGE's context lengths, GQA's main gift is inference speed / bandwidth, not survival — but take it anyway; it is nearly free.**

---

## 3. Tokenizer recommendation for code

### Recommendation (build this)

> **Train a byte-level BPE (BBPE) tokenizer from scratch on a representative sample of FORGE's final training mix (code-heavy, Python-first, plus the English instructions / docstrings the model must read). Vocab: 16,384 for the ~125M model; 32,768 for the ~350M model. Tie input/output embeddings. Reserve explicit special tokens from day one. Validate with fertility + downstream probes before any long training run.**

### 3.1 BPE vs. byte-level vs. character — the actual trade-off

| Scheme | Vocab | Sequence length for same code | Pros | Cons for FORGE |
|---|---|---|---|---|
| Pure byte-level | 256 | Longest (~3–4× BPE on English/code; exact factor is corpus-dependent) | Zero OOV, tiny embedding, perfect for weird Unicode / identifiers | Attention is O(n²): 3× longer = ~9× attention compute. Training steps are wasted re-learning spelling. No published strong small coder uses it as the *sole* scheme. |
| Character-level | ~100–1k | Long, similar problem | Simple | Same length problem; worse than bytes on Unicode. |
| **Byte-level BPE (BBPE)** | **16k–32k for FORGE** (see §3.2) | **Short** — frequent code patterns become single tokens | Byte fallback ⇒ zero OOV (any string decomposes to bytes); compression where it matters; this is what Qwen (BBPE, 151,643 regular tokens for the *general* Qwen2.5 tokenizer), GPT-2-style, StarCoder-family, and Llama-family tokenizers converge on | Must be trained on the right corpus (see §3.3). Rare tokens in an oversized vocab are under-trained. |
| Word-level | 100k+ | Shortest for seen words | — | OOV catastrophe on identifiers (`DataLoaderConfigOverride` has never been seen before and never will be again). Never for code. |
| AST / structure-aware tokens (`<INDENT>`, `<DEDENT>`, normalised literals) | BPE + specials | Shorter, more regular | Historically explored (e.g. TabNine-era work); can help a tiny model by removing surface noise | Breaks compatibility with every standard tool, makes generated code need detokenisation surgery, and modern BBPE models learn indentation fine without it. **[Hypothesis — FORGE to test only if BBPE baseline shows a specific indentation failure mode]** |

Sources: Qwen2.5 BBPE + vocab description: https://arxiv.org/html/2412.15115v2 · General BPE / vocab-size trade-off exposition: https://imaddabbura.github.io/posts/nlp/BPE-Tokenizer.html

### 3.2 Why vocab size is a *parameter-budget* decision at FORGE's scale [Established — arithmetic]

Embedding parameters = `vocab × d_model` (halved in effect by tying — one matrix instead of two — but not zero).

| Config | d_model | Vocab | Embedding params | Share of total budget |
|---|---:|---:|---:|---:|
| FORGE-125M candidate | 768 | 16,384 | 12.6M | ~10% of 125M — healthy |
| FORGE-125M with a copied 50k vocab | 768 | 50,257 | 38.6M | ~31% — a third of the model spent on rare-token embeddings |
| FORGE-350M candidate | 1,024 | 32,768 | 33.6M | ~10% of 350M — healthy |
| Qwen2.5-Coder-0.5B (for contrast) | 896 | 151,646 | ~135.9M | ~27% of 0.5B — Qwen *accepts* this because its vocab serves 92 languages + general text, and it has 5.5T tokens to train those embeddings. FORGE has neither reason. |

Qwen2.5-Coder config table (hidden sizes, layers, Q/KV heads, tying ✓ for 0.5B/1.5B/3B, vocab 151,646, 5.5T trained tokens): https://ar5iv.labs.arxiv.org/html/2409.12186

**Rule of thumb for FORGE:** keep embeddings ≤ ~15% of total parameters. Every token added to the vocab must "pay rent" by being frequent enough in FORGE's corpus to (a) shorten sequences measurably and (b) receive enough gradient updates to be well-trained. With FORGE-scale corpora (millions to low billions of tokens, not trillions), most of a 100k+ vocab would be neither.

**Established supporting pattern:** TinyLlama (1.1B, general) uses the Llama-2 32k vocab: https://arxiv.org/html/2401.02385v2 · SmolLM2's small models use a ~49k custom vocab and its 1.7B model uses tied embeddings: https://arxiv.org/pdf/2502.02737 · TinyStories deliberately cut its GPT-Neo tokenizer to the top 10k tokens for a tiny-vocab domain: https://ar5iv.labs.arxiv.org/html/2305.07759

### 3.3 Code-specific requirements (non-negotiable in the FORGE tokenizer)

Drawing on how code breaks prose tokenizers [Established failure modes]:

1. **Whitespace is syntax in Python.** The tokenizer must losslessly preserve runs of spaces, tabs, and newlines, and ideally allocate single tokens for the most common indentation patterns (e.g. 2, 4, 8 spaces — as the StarCoder tokenizer is described as doing). A tokenizer that merges or normalises whitespace inconsistently (4 spaces = 1 token here, 2+2 there) forces the model to spend capacity re-deriving block structure. Sources: https://llm.hosseini.work/book/part5/training_tokenization.html · https://llmbook.icsgen-ai.org/part-1-llm-building-blocks/module-01-foundations-nlp-text-representation/section-1.5.html
2. **Multi-character operators must be atomic or near-atomic:** `==`, `!=`, `<=`, `>=`, `->`, `::`, `**`, `//`, `+=`, `...`. If BBPE is trained on enough code, these merges are learned automatically — *verify* they exist in the trained vocab; do not assume.
3. **Identifiers split at subword boundaries the model can recompose.** `snake_case` and `camelCase` names will split (e.g. `get`, `User`, `Prefer`, `ences`) — that is acceptable and even helpful for generalisation, provided splits are *consistent*. Inconsistent splitting of the same identifier is the failure to test for.
4. **Reserve special tokens now; adding them later requires embedding surgery:** at minimum `<|endoftext|>` / EOS, PAD, `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|fim_pad|>` (so FIM is possible later without a vocab change — Qwen2.5-Coder and StarCoder2 both train with exactly this FIM token family), plus `<|repo_name|>` and `<|file_sep|>` for future repo-level data. Qwen's special-token table: https://ar5iv.labs.arxiv.org/html/2409.12186
5. **Numbers:** BPE splits numbers by corpus frequency, not place value, which is a known contributor to arithmetic weakness. For *code generation* (as opposed to arithmetic) this is a second-order concern — do not adopt digit-by-digit tokenisation for v1; log it as a known limitation. **[Established problem; FORGE mitigation deferred deliberately]**
6. **Measure fertility before training:** tokens-per-byte (or tokens-per-line) on (a) held-out Python, (b) held-out English instructions, (c) any second language FORGE targets. Compare FORGE's tokenizer against (i) GPT-2 BBPE and (ii) Qwen's tokenizer on the same files. If FORGE's tokenizer is not *better on FORGE's corpus* than an off-the-shelf one, there is no case for training it — reuse one and spend the saved cycles on data. This is a cheap, decisive pre-experiment.

### 3.4 What not to do

- Do not train the tokenizer on a different distribution from the model corpus (e.g. tokenizer on raw GitHub, model on filtered textbooks) — fertility and merge quality follow the *tokenizer's* corpus.
- Do not use a SentencePiece Unigram tokenizer merely because Llama-1 did; every current code family FORGE is learning from uses BPE/BBPE, and HF / llama.cpp tooling paths are smoothest there. (This is a tooling recommendation, not a claim that Unigram is worse in principle.)
- Do not freeze the vocab before inspecting the 500 most frequent tokens and 200 random mid-frequency tokens by eye. Tokenizer bugs are silent — the model trains, loss falls, and capability is quietly capped.

---

## 4. Key efficiency techniques worth adopting for a from-scratch small model

Ordered by (value × certainty) ÷ implementation cost for FORGE. "Adopt" = in v1. "Ablate" = implement behind a config flag and earn its place. "Defer" = not in v1.

### Adopt in v1

| Technique | What it does | Evidence | FORGE setting |
|---|---|---|---|
| **Pre-norm + RMSNorm** | Normalise the *input* of each sub-layer (not the output); RMSNorm drops LayerNorm's mean-centring, keeping only RMS rescaling — cheaper, equally stable in Llama-family ablations | Introduced by Zhang & Sennrich (2019); adopted by Llama, TinyLlama ("pre-norm … to stabilise training; RMSNorm to improve training efficiency"), Qwen, SmolLM2 [Established]. TinyLlama: https://arxiv.org/html/2401.02385v2 · Llama architecture summary: https://hivam.org/blog/posts/06-llama-model.html | RMSNorm, eps 1e-5/1e-6, pre-norm, no bias in norm |
| **RoPE (Rotary Position Embedding)** | Rotates Q/K vectors by a position-dependent angle at *every* layer; attention scores then depend on *relative* distance. No learned positional parameters, graceful length behaviour | Su et al., RoFormer: RoPE "encodes absolute position with a rotation matrix and meanwhile incorporates explicit relative position dependency", with length flexibility and decaying inter-token dependency: https://arxiv.org/abs/2104.09864v4 · Used by TinyLlama, Qwen, DeepSeek-Coder, StarCoder2 [Established] | RoPE on full head-dim, θ = 10,000 at 1k–2k context (raise θ if context is later extended; do not copy Llama-3's θ=500,000 blindly — it is tuned for 128k) |
| **SwiGLU feed-forward** | `down(SiLU(gate(x)) ⊙ up(x))` — a gated MLP with 3 matrices instead of 2; consistently better quality-per-parameter than ReLU/GELU MLPs in the PaLM/Llama lineage | Shazeer (2020) GLU variants, via Llama; TinyLlama: "instead of the traditional ReLU … we adopt SwiGLU" https://arxiv.org/html/2401.02385v2 [Established] | Intermediate size ≈ 8/3 × d_model, rounded to a multiple of 256 for kernel efficiency (TinyLlama: 2048 → 5632) |
| **Grouped-Query Attention (GQA)** | Partition Q heads into groups sharing one K head + one V head. KV cache and K/V projection params shrink by the grouping factor; quality stays close to full MHA, speed approaches MQA | Ainslie et al. (2023): GQA is "a generalisation of multi-query attention which uses an intermediate number of key-value heads"; "uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA"; MQA alone "can lead to quality degradation": https://arxiv.org/abs/2305.13245v2 and full text https://arxiv.org/html/2305.13245v3 [Established] | Real configs to copy the *ratio* from, not the counts: TinyLlama 32 Q heads / 4 KV groups; Qwen2.5-Coder-0.5B 14 Q / 2 KV, 1.5B 12 Q / 2 KV, 3B 16 Q / 2 KV (https://ar5iv.labs.arxiv.org/html/2409.12186). FORGE: grouping factor 4 (e.g. 12 Q / 3 KV or 16 Q / 4 KV) is a sane default; factor 6–7 (Qwen-small style) is an ablation. Avoid MQA (1 KV head) for v1 — the quality/instability risk is documented and FORGE has no need to pay it at 2k context. |
| **Tied embeddings** (input embedding = output LM head) | Halves embedding storage, regularises rare tokens (their output vector is trained whenever they appear as input) | Qwen2.5-Coder ties embeddings for 0.5B/1.5B/3B and unties at 7B+ (table in §3.2 source); SmolLM2-1.7B tied; MobileLLM's "embedding sharing" is a headline technique for sub-billion models [Established for <~3B] | Tie for all FORGE sizes in this brief (≤350M, and even at 0.5–1.5B if FORGE goes there). Untie only as a measured ablation at ≥1B. |
| **No / minimal bias, BF16, FlashAttention-2** | Bias-free linear layers (Llama style) save trivial params and match the reference implementations FORGE will copy; BF16 + FlashAttention-2 cut memory and raise throughput on Kaggle T4/A100-class GPUs without changing the maths materially | FlashAttention-2 + fused RMSNorm/SwiGLU/RoPE are how TinyLlama reached its reported 24k tok/s on A100-40G [Reported]; DeepSeek-Coder likewise uses FlashAttention-2 [Reported] | Use PyTorch SDPA / FlashAttention where available with a correct fallback path for Termux/CPU debugging (results must agree within tolerance — test it). Qwen's QKV bias is the documented exception to "no bias"; FORGE follows Llama (no bias) for v1 simplicity. |

### Ablate (implement, measure, keep only if it wins)

| Technique | Hypothesis for FORGE | How to test cheaply |
|---|---|---|
| **Deep-and-thin shape** (more layers, smaller d_model at fixed params) | MobileLLM's sub-billion gains transfer from commonsense benchmarks to *code* [Hypothesis — the paper's headline suite is not code] | Train 125M in two shapes (e.g. 12×768 vs. 24×512-ish, matched params ±5%) on identical data / token budget; compare held-out loss *and* syntax-valid rate + pass@1 on FORGE's dev problems. Pre-register: adopt deep-and-thin only if it wins on pass@1, not just loss. |
| **Block-wise weight sharing** (MobileLLM-LS: repeat a block, sharing its weights) | Saves stored params at small latency cost; reported +0.7%/+0.8% over MobileLLM 125M/350M on their suite [Reported, non-code] | Same protocol as above. Especially relevant if FORGE's binding constraint becomes *storage* rather than compute — on most phones it is not, so prior is sceptical. |
| **FIM (fill-in-the-middle) training** at ~50% rate, PSM format | StarCoder2 and DeepSeek-Coder both train with FIM and report strong infilling; DeepSeek's ablation found 50% PSM balances infilling vs. left-to-right completion, 100% FIM maximises infilling but *weakens* completion [Reported] | Only after the left-to-right baseline works. Measure both HumanEval (completion) and a HumanEval-FIM-style probe; adopt only if completion does not regress beyond a pre-set margin. Requires the FIM special tokens reserved in §3.3 — which is why they are reserved now. |
| **Sliding-window attention** | Caps KV cache for long files [Established mechanism; value at FORGE's context unproven] | Defer until context >4k is on the roadmap (see §1.1). |
| **Curriculum / textbook-style ordering** (concepts before compositions; exercises after exposition) | Phi-1's data *design* (textbooks + exercises), separated from its teacher, transfers to FORGE's legally-usable data [Hypothesis] | Two data orderings / mixes, identical model + token budget. This is a *data* ablation run through the architecture pipeline — likely FORGE's highest-expected-value ablation of all. |

### Defer / reject for v1

- **MQA (1 KV head):** documented quality degradation / instability risk (GQA paper) for a saving FORGE does not need at ≤2k context.
- **MoE:** storage is total params (see §1.1).
- **MLA, linear attention, pure SSM:** see §1, Rank 4.
- **Speculative decoding, Medusa heads, etc.:** inference-time tricks for a model that does not exist yet. llama.cpp already provides a sane deployment path; optimise the model first.
- **Quantisation-aware training (QAT):** post-training Q4_K_M via llama.cpp is the v1 deployment route (perplexity cost of Q4_K_M is modest in reported mobile benchmarks, §2.4). QAT / SpinQuant-style methods showed large reported speedups for Llama 3.2 1B on device, but they are a *later* optimisation once FORGE has a checkpoint worth optimising.

### 4.1 Training-efficiency notes that are not architecture (but decide whether FORGE trains at all)

- **Overtraining small models is the modern norm, and FORGE cannot afford the modern norm.** Chinchilla-style compute-optimal training (~20 tokens/param) has been deliberately abandoned by TinyLlama (3T tokens / 1.1B ≈ 2,700 tokens/param), SmolLM2 (11T / 1.7B), and Qwen2.5-Coder (5.5T across sizes) in favour of inference-efficient, heavily-overtrained checkpoints [Reported token counts in sources above]. FORGE's free-GPU budget will land far on the *undertrained* side of even Chinchilla for 350M (≈7B tokens for Chinchilla-optimality). Therefore: **(a)** prefer the 125M size for from-scratch science (Chinchilla ≈ 2.5B tokens — still hard, but imaginable across many Kaggle sessions with resumable checkpoints), **(b)** treat data quality / repetition of high-quality data (Phi-1 trained ~8 passes over its 7B tokens [Reported]) as FORGE's substitute for raw token count, while monitoring overfitting on a held-out set, and **(c)** never report FORGE's loss / pass@1 next to Qwen's without stating both token counts in the same sentence.
- **Checkpointing and resumability are architecture-adjacent requirements:** free sessions die. Save optimiser state, use config files + fixed seeds, and never overwrite the last known-good checkpoint (project requirement). A training loop that cannot resume is, for FORGE, a failed design regardless of its loss curve.
- **Free-tier limits were not re-verified for this brief.** Kaggle / Colab GPU types, weekly quotas, session lengths, and disk sizes change and differ by account. Any FORGE plan that names a specific quota must verify it at execution time against the platform's current official page — do not copy a quota number from a blog (including this brief, which deliberately states none).

---

## 5. Proposed first architecture (the concrete bet)

One codebase, two configs. Both are Rank-1 Llama-style decoder-only Transformers differing only in size — so the 30M/125M experiments de-risk the 350M run instead of being throwaway work.

| Field | FORGE-125M (first real experiment) | FORGE-350M (first usefulness test) |
|---|---|---|
| Layers | 12 (ablate 24 thin variant) | 24 |
| d_model | 768 | 1,024 |
| Q heads / KV heads (GQA) | 12 / 3 (factor 4), head-dim 64 | 16 / 4 (factor 4), head-dim 64 |
| MLP | SwiGLU, intermediate 2,048 | SwiGLU, intermediate 2,816 |
| Vocab / tokenizer | 16,384 BBPE, code-trained (§3) | 32,768 BBPE, code-trained (§3) |
| Embeddings | Tied | Tied |
| Norm / position | RMSNorm pre-norm / RoPE θ=10,000 | Same |
| Context (train) | 1,024 → 2,048 in a second phase | 2,048 |
| Special tokens | EOS/PAD + FIM family + repo/file separators reserved | Same tokenizer family, retrained at 32k |
| Precision / attention | BF16, FlashAttention-2 / SDPA with CPU fallback | Same |
| Deployment target | GGUF Q4_K_M, ~80 MB | GGUF Q4_K_M, ~200 MB |
| Role | Ablations: tokenizer, depth/width, data mix, LR. Success = reproducible pipeline + honest dev pass@1, *not* a headline score. | Usefulness probe: syntax validity, unit-test pass rate, and dev pass@1 vs. the 125M and vs. a fine-tuned Qwen2.5-Coder-0.5B baseline in the *same* harness. |

Parameter counts above are *targets* (±10% once embeddings / exact intermediate rounding are finalised in code) — the implementation milestone must print the true count from the instantiated model and record it in the config, not copy this table.

**Why this bet, in one paragraph:** it is the architecture with (1) the strongest published small-code results per parameter (Phi-1 / Qwen2.5-Coder lineage), (2) the cheapest path to phone deployment (llama.cpp GGUF), (3) the most free debugging knowledge, and (4) clean ablation surfaces (one config flag each for GQA factor, tying, depth/width, FIM). Its known weakness — it will be undertrained on FORGE's compute — is a *data and budget* problem that no alternative architecture in §1 fixes.

---

## 6. Open questions FORGE must answer experimentally (not by citation)

1. Does deep-and-thin beat wide-and-shallow *on code pass@1* at 125M under FORGE's token budget? (MobileLLM evidence is non-code.)
2. What is FORGE's tokenizer fertility win, if any, over Qwen's tokenizer on FORGE's exact corpus? If ≤5%, reuse wins.
3. How many epochs over a small, execution-verified Python corpus can FORGE sustain before held-out pass@1 degrades? (Phi-1's ~8 passes are an existence proof for *its* data, not a FORGE constant.)
4. At what point does execution-filtered SFT data beat more pretraining tokens, hour-for-hour on a free GPU? (The base→instruct gaps in §2.2 suggest "early" — measure it.)
5. What is FORGE's *measured* tok/s, peak RSS, and thermal behaviour on the actual target phone in Termux / llama.cpp? No number in §2.4 substitutes for this measurement.

---

## Sources (primary first)

**Technical reports / papers**
- Qwen2.5-Coder Technical Report — architecture table, base & instruct HumanEval/MBPP (EvalPlus): https://arxiv.org/pdf/2409.12186 · HTML: https://arxiv.org/html/2409.12186v2 · ar5iv: https://ar5iv.labs.arxiv.org/html/2409.12186
- Qwen2.5 Technical Report — BBPE tokenizer (151,643 regular tokens), tying by size, GQA at all sizes: https://arxiv.org/html/2412.15115v2
- StarCoder2 and The Stack v2 — Table 9 base-model pass@1 (greedy), GQA + sliding window + FIM: https://arxiv.org/html/2402.19173v1
- DeepSeek-Coder — 1.3B–33B, 2T tokens, FIM ablation (50% PSM balance): https://arxiv.org/pdf/2401.14196 · HTML: https://arxiv.org/html/2401.14196v1
- Textbooks Are All You Need (Phi-1) — 1.3B, 50.6% HumanEval / 55.5% MBPP, 350M small model, textbook + exercise data design: https://arxiv.org/pdf/2306.11644 · HTML: https://ar5iv.labs.arxiv.org/html/2306.11644
- TinyLlama — 1.1B, Llama-2 architecture, RMSNorm / SwiGLU / RoPE / GQA (32 Q heads, 4 KV groups), 3T tokens: https://arxiv.org/html/2401.02385v2 · PDF: https://arxiv.org/pdf/2401.02385v1
- TinyStories — sub-10M models, narrowed synthetic domain, 10k-token vocab cut: https://arxiv.org/pdf/2305.07759 · ar5iv: https://ar5iv.labs.arxiv.org/html/2305.07759
- MobileLLM — sub-billion architecture, deep-and-thin, embedding sharing, GQA, block-wise sharing gains: https://proceedings.mlr.press/v235/liu24ce.html
- SmolLM2 — data-centric small-model training, tied embeddings, RoPE, SwiGLU configs: https://arxiv.org/pdf/2502.02737
- GQA — Ainslie et al. 2023: https://arxiv.org/abs/2305.13245v2 · Full text: https://arxiv.org/html/2305.13245v3
- RoFormer / RoPE — Su et al. 2021: https://arxiv.org/abs/2104.09864v4
- SSM vs. Transformer long-context characterisation (Mamba-2 crossover ~32k; hybrid results): https://arxiv.org/html/2507.12442v4

**Secondary / device-measurement sources (labelled as such wherever used)**
- Phi-3 family numbers as summarised from Microsoft's reports (verify against the original report before publication use): https://systems-analysis.info/eng/Phi_%28Microsoft%29
- Qwen2.5-Coder-1.5B Q4 size notes: https://localaimaster.com/models/qwen-2-5-coder-1-5b
- On-device KMP + llama.cpp benchmarks (3B Q4_K_M, Pixel 8 / iPhone 15 Pro): https://dev.to/software_mvp-factory/on-device-llm-inference-via-kmp-and-llamacpp-4pec
- llama.cpp Android notes: https://deepwiki.com/max-krasnyansky/llama.cpp/8.1-android
- ExecuTorch Llama 3.2 on OnePlus 12 (project README measurements): https://android.googlesource.com/platform/external/executorch/+/HEAD/examples/models/llama/README.md
- Tokenisation background (BPE trade-offs, code whitespace / indentation handling): https://imaddabbura.github.io/posts/nlp/BPE-Tokenizer.html · https://llm.hosseini.work/book/part5/training_tokenization.html · https://llmbook.icsgen-ai.org/part-1-llm-building-blocks/module-01-foundations-nlp-text-representation/section-1.5.html
- Llama architecture exposition (RMSNorm / SwiGLU / RoPE placement): https://hivam.org/blog/posts/06-llama-model.html

*End of brief. Next artefacts that depend on this document: dataset strategy (licence + decontamination), benchmark harness specification (single pinned EvalPlus-style harness), and the FORGE-125M implementation milestone.*
