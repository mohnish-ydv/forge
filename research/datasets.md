# FORGE — Dataset Strategy Brief

> Scope: Training a small, genuinely capable coding model from scratch on ₹0 budget — Android / Termux for development, Kaggle + Colab free GPUs for training. All sizes / licences below are from dataset cards, papers, or official docs checked 2026-10-24. Where a source is a secondary summary, it is flagged. Do **not** treat any number here as a guarantee — re-check the live card before downloading.

## TL;DR Recommendation

**Do not download The Stack v2 raw.** Start with already-filtered, openly downloadable derivatives, stream them, and spend scarce GPU time on training — not on re-cleaning 67 TB:

1. **Primary pretraining:** `HuggingFaceTB/stack-edu` — Stack v2 filtered by educational-quality classifiers (score ≥3). Python-first slice for v0.
2. **Python bootstrap / ablation baseline:** `HuggingFaceTB/python-edu` (as used in SmolLM corpus) — ~4B tokens of educational Python from The Stack.
3. **Natural-language / docstring grounding (10-15% mix):** `HuggingFaceFW/fineweb-edu` sample-10BT / sample-100BT — ODC-By.
4. **SFT (after pretraining, not instead of it):** `ise-uiuc/Magicoder-OSS-Instruct-75K` (MIT) + execution-filtered self-generated problems (see §5).
5. **Hold-out for curriculum / annealing:** OpenCoder `opc-annealing-corpus` idea — algorithmic + synthetic — replicated at small scale, not downloaded wholesale at v0.

Language scope for v0: **Python only, then Python + JavaScript/TypeScript + Rust/Go in v1.** Multilingual from day 1 dilutes a <500M model.

---

## 1. Recommended Datasets — Licences and Sizes

### 1.1 Core code corpora

| Dataset (HF id) | Size (as published) | Licence / Access | Suitability for FORGE |
|---|---|---|---|
| **The Stack v1** `bigcode/the-stack` | Full 6.4 TB [1] | Collection of permissively-licensed files; use must abide by original licences. Card notes >3 TB of permissively-licensed files in 30 languages in early description, expanded in v1 releases. | Historical baseline only. Superseded by v2. Still useful because v1 files are directly downloadable vs v2 SWH-ID indirection — but do not start new training on v1 without dedup/decontamination. |
| **The Stack v2** `bigcode/the-stack-v2` | Full **67.5 TB**, dedup **32.1 TB**, train-full **~900B tokens**, train-smol **600B+ unique tokens**, **658 languages**, **3.28B unique files from 104.2M repos** [2] | **Not a single licence.** "A collection of source code from repositories with various licenses. Any use must abide by the terms of the original licenses" [2]. Files are permissive or `no_license` per card. **Gated:** must accept Terms, share HF contact info, and bulk download requires agreement with Software Heritage + INRIA. HF repo stores only SWHIDs, not file contents — contents fetched from Software Heritage S3 [2]. | **Reference corpus, not a direct download for FORGE.** The gating + SWH fetch + 67.5 TB makes it impractical on phone/Kaggle disk. Use its filtered derivatives (Stack-Edu, train-smol-ids list) instead. StarCoder2 was trained on v2.0.1 [2]. |
| **The Stack v2 dedup** `bigcode/the-stack-v2-dedup` | 32.1 TB deduped from 67.5 TB [2] | Same per-file licence regime as v2. | Same access problem as v2. Prefer Stack-Edu which builds on it. |
| **Stack-Edu** `HuggingFaceTB/stack-edu` | Filtered Stack v2 across 15 languages. Classifier paper table (SmolLM2): before → after filtering (B tokens): Python 50.6 → **21.8**, C++ 69.7 → **16.0**, JavaScript 45.3 → **11.1**, Java 45.6 → **42.1** (Java used threshold 2, others 3) [3]. Another training-mix listing cites Stack-Edu at 86.7B tokens in that specific mix [4] — total depends on thresholds / language subset, so quote per-language numbers, not a single total. Viewer shows fields: `blob_id, language, repo_name, path, score, int_score, detected_licenses, license_type` [5]. | Inherits Stack v2 per-file licences (`permissive` / `no_license` in `license_type`). Filter additionally by `detected_licenses` if you need strict permissive-only (see §3). | **Best FORGE starting point.** Educational filtering demonstrably helped: in SmolLM2 ablations, filtering with threshold 3 improved MultiPL-E for Python 20.7 → 25.6, C++ 16.7 → 24.8, JS 18.2 → 22.4 [3]. Small enough to stream a single language. |
| **Python-Edu** (SmolLM Corpus component) | **4B tokens** of educational Python samples from The Stack [6] | Part of SmolLM Corpus (Cosmopedia v2 28B + Python-Edu 4B + FineWeb-Edu) [6]. Underlying code is Stack permissive subset. Check the `HuggingFaceTB/smolLM-corpus` card for redistribution terms before mirroring. Direct `python-edu` HF path returned gated/401 in this check — use the smolLM-corpus release path, not a guessed URL. | **Ideal v0 prototype corpus.** 4B tokens is exactly the scale FORGE can actually finish (see §6). If access blocks, fall back to Stack-Edu Python slice filtered `int_score >= 3`, which replicates the recipe. |
| **OpenCoder pretraining corpora** `open-coder/*` via OpenCoder project | Pretraining total **2.5T tokens** (90% raw code + 10% code-related web) for the full project [7]. Components listed in repo: `fineweb-code-corpus` **148 GB**, `fineweb-math-corpus` **10 GB**, `opc-annealing-corpus` **24 GB** [8]. SFT: **4.5M examples** total (Stage 1 4.21M + Stage 2 375K in DeepWiki summary of repo README) [8]. | OpenCoder models state "support commercial applications under a permissive License" with a LICENSE file in the model repo [7]. **Dataset licences are per-component** — verify each HF dataset card separately before use; do not infer dataset licence from model licence. | Too large for v0 pretraining, but **the pipeline is the template to copy**: file- + repo-level dedup, RefineCode filtering, and a small high-quality annealing corpus at the end of training [8]. The 24 GB annealing idea is directly reusable at 100-500 MB scale for FORGE. |
| **CodeParrot** `codeparrot/codeparrot` (historical) | Python GitHub code, ~180 GB class dataset used as baseline in Stack paper | Apache-2.0 for the dataset code; underlying files per GitHub repos. | Only as a **negative baseline**: Stack paper models trained on Stack-Dedup Python got HumanEval pass@1 19.5% vs 3.7% for CodeParrot baseline in that study [9]. Shows why FORGE should not start from unfiltered GitHub scrapes. |

[1] Stack v2 card comparison table: v1 full 6.4 TB vs v2 full 67.5 TB. HF `bigcode/the-stack-v2`.
[2] HF `bigcode/the-stack-v2` card — size table, gating terms, SWHID-only structure, licence discussion, 3.28B files / 104.2M repos / 67.53 TB uncompressed.
[3] HF `HuggingFaceTB/stack-edu-classifier-*` cards — Llama-3.1-annotated 500K pairs, StarEncoder classifier, threshold experiments and per-language before/after table.
[4] VAETKI model card training table listing Stack-Edu 86.7B in its 9.8T mix — illustrates mix-dependent totals.
[5] HF `HuggingFaceTB/stack-edu` dataset viewer schema.
[6] HF `HuggingFaceTB/SmolLM-360M` / `SmolLM-135M` model cards: SmolLM Corpus = Cosmopedia v2 (28B) + Python-Edu (4B) + FineWeb-Edu (220B in that corpus description).
[7] HF `infly/OpenCoder-1.5B-Base` and `OpenCoder-8B-Instruct` cards: 2.5T tokens, 90/10 split, 4.5M SFT.
[8] OpenCoder-llm DeepWiki summary of README: component sizes and processing flow.
[9] AlphaXiv overview of The Stack paper (arXiv:2211.15533): dedup +33% relative HumanEval gain, CodeParrot comparison table.

### 1.2 Web / educational text (for docstrings, instructions, reasoning — not as main code source)

| Dataset | Size | Licence | Use in FORGE |
|---|---|---|---|
| **FineWeb** `HuggingFaceFW/fineweb` | Originally **15T tokens**, current card headlined **>18.5T tokens** of cleaned, deduplicated English web from CommonCrawl (96 dumps, 2013-2024 in paper) [10][11] | **ODC-By 1.0** for the dataset release [10][11]. PII step in paper: anonymising emails and public IPs [11]. | Use only via **FineWeb-Edu**, not raw FineWeb. Raw web dilutes code ability per token for a tiny model. |
| **FineWeb-Edu** `HuggingFaceFW/fineweb-edu` | **1.3T token** educational subset of FineWeb in paper [11]; practical subsets: `sample-10BT`, `sample-100BT` (100B GPT-2 tokens, 1823 parquet shards ~100 MB each in nanochat repackaging) [12] | ODC-By 1.0 (inherited from FineWeb). | **10BT sample is the FORGE-sized version.** Mix 10-15% to keep English instruction / docstring understanding without spending the budget on web text. Filter to code-adjacent pages if possible (OpenCoder's FineWeb code recall idea). |

[10] HF `HuggingFaceFW/fineweb` card + search summary: 15T originally, 18.5T current, ODC-By 1.0, DataTrove pipeline.
[11] FineWeb paper arXiv:2406.17557: 15T from 96 CommonCrawl snapshots, per-crawl MinHash dedup, C4 + custom filters, PII anonymisation, FineWeb-Edu 1.3T.
[12] nanochat DeepWiki dataset page for `karpathy/fineweb-edu-100b-shuffle`: 100B tokens, 1823 shards.

### 1.3 Instruction / SFT datasets (use AFTER pretraining)

| Dataset | Size | Licence (as listed on card / paper citing it) | Notes |
|---|---|---|---|
| **Magicoder-OSS-Instruct-75K** `ise-uiuc/Magicoder-OSS-Instruct-75K` | 75K problems/solutions generated via OSS-Instruct from open-source snippets using `gpt-3.5-turbo-1106` [13] | **MIT** (per arXiv paper citing the dataset: "Magicoder-OSS-Instruct-75K (MIT License)") [14] — **caveat:** data was generated with a proprietary model; check current HF card terms and OpenAI-derived-data policy for your intended distribution. Do not assume MIT on the dataset erases teacher-model terms. | Low-bias synthetic instructions grounded in real code snippets [13]. Good SFT seed, but **must be execution-filtered** — not all solutions pass tests (see §5). |
| **Magicoder-Evol-Instruct-110K** `ise-uiuc/Magicoder-Evol-Instruct-110K` | 110K, decontaminated redistribution of evol-codealpaca-v1 [13] | **Apache-2.0** per same citing paper [14] | Use second, after OSS-Instruct, replicating Magicoder's 2+2 epoch recipe at small scale — or skip if it hurts held-out eval. |
| **StarCoder training mix** (for reference, not a single download) | StarCoderBase: **1T tokens**, 80+ languages, plus Git commits, GitHub issues, Jupyter notebooks; StarCoder = Base + **35B Python tokens** further training [15]. StarCoder2: **3.3-4.3T tokens** on Stack v2, 600+ languages, 3B/7B/15B sizes [16] | StarCoder models under **BigCode OpenRAIL-M** (permits commercial use under responsible-AI conditions) [16]; data is Stack (per-file licences). | Do not try to replicate 1T tokens. The *techniques* to copy: FIM objective, notebook handling (code-only via Jupytext + code-text pairs), anonymised issues, commit windowing [15]. |

[13] HF Magicoder model cards: OSS-Instruct and Evol-Instruct descriptions and provenance.
[14] arXiv:2407.05040 citing both Magicoder datasets with licence labels and HF URLs.
[15] AlphaXiv / StarCoder paper arXiv:2305.06161 summary: Stack v1.2, 1T tokens, 86 languages selected, MinHash+LSH 5-grams, Jaccard 0.7.
[16] Secondary summary of StarCoder2 (mstone.ai) — treat sizes as indicative; verify on HF `bigcode/starcoder2-*` cards before quoting in a paper.

### 1.4 What NOT to use at v0

- **The Pile / RedPajama full, Dolma full, FineWeb full** — web-scale, not code-focused, cannot be stored or shuffled on free disk.
- **Unfiltered GitHub scrapes (own crawl)** — you will spend all your time on licence detection, dedup, and malware/PII, and still underperform Stack-Edu. The Stack project already did this with ScanCode + GHArchive [2].
- **Any dataset with unclear licence or "research only / non-commercial" without checking** — FORGE aims for a clean open-source repo. When in doubt, keep the dataset out of the redistributed training mix and document it as "weights-only experiment."
- **Stack v3 claims circulating in 2026 blogs** (~5T tokens, 114 TB, July 2026) [secondary blogs] — **not verified on an official HF card in this check.** Do not base FORGE on v3 until `bigcode/the-stack-v3` (or equivalent) appears with a card, licence, and access terms you have read yourself.

---

## 2. Free-Tier Limits — Verified, Not Assumed

> Free tiers change without notice and are quota-managed dynamically. The numbers below are the best public evidence on 2026-10-24; always read the quota banner in your own account before a run.

| Platform | What free actually gives (evidence) | Hard limits that shape FORGE |
|---|---|---|
| **Kaggle Notebooks** | **~30 GPU hrs/week**, sometimes more depending on demand; **1× P100 or 2× T4**; **TPU v3-8 up to ~20 hrs/week**; sessions **up to 12 hrs CPU/GPU, ~9 hrs TPU**; ~29-30 GB RAM (GPU session), 20 GB auto-saved `/kaggle/working` disk in one summary, larger ephemeral disk reported elsewhere — plan for **20 GB persistent** [17][18][19] | Most predictable free GPU. Weekly wall-clock quota counts idle time [18]. GPU needs phone verification [19]. No paid top-up. Datasets capped at 200 GB each per summary [17]. **Design implication:** checkpoint every 30-60 min to `/kaggle/working`, push to HF Hub / GitHub release, never rely on a single 12-hr session finishing a run. |
| **Google Colab Free** | **No published quota.** Google "deliberately does not publish GPU type, idle timeout, or VM lifetime, and states they vary" [20]. Typical allocation reported across 2025-26 guides: **T4 16 GB**, **max ~12 hrs/session**, pre-emptible, not guaranteed [21][22]. RAM ~12-15 GB free. | Treat Colab as **opportunistic extra compute**, not the schedule. Use it for tokeniser training, data filtering, eval, and short training bursts. Save to Google Drive / HF Hub continuously. Background execution is paid-only. |
| **Hugging Face Hub** | Dataset/model hosting and `datasets` streaming are free for public repos (with fair-use / storage limits that change — check HF storage docs at upload time). **ZeroGPU Spaces:** free accounts **5 min GPU/day** (2 min unauthenticated, 40 min PRO) on current hardware per 2026 roundup [18] — Gradio-only, not training. | Use HF as **FORGE's checkpoint + dataset store**: stream with `load_dataset(..., streaming=True)`, push checkpoints as model repos. Do not plan training on ZeroGPU. Gated datasets (Stack v2) require manual agreement in the browser — cannot be automated from Termux alone. |
| **Android / Termux (dev device)** | No GPU training. CPU, storage, and RAM vary by phone; assume ≤8 GB RAM, slow eMMC/UFS, thermal throttling. | Termux is for: writing code, running the cleaning pipeline on **samples** (1-10K files), tokeniser experiments, unit tests, and orchestrating Kaggle/Colab via API/CLI. Never attempt full-corpus dedup on the phone. |

[17] AgentDeals Kaggle listing citing `kaggle.com/docs/efficient-gpu-usage` read 2026-10-23: 30 hrs, P100/T4x2, TPU 20 hrs, 12-hr sessions, disk/RAM figures.
[18] Spheron 2026 free-GPU roundup: Kaggle ~30 GPU hrs/week, quota resets Sunday 00:00 UTC, counts wall-clock; HF ZeroGPU 5 min/day free; Colab notes.
[19] AIKetra / Kaggle community summaries 2026: ~30 GPU / 20 TPU hrs/week, phone verification, queueing.
[20] Spheron roundup § Colab: 12-hr max, unpublished GPU type/idle/VM lifetime.
[21] ThunderCompute Colab tiers summary: free assigns shared T4, up to 12 hrs, not guaranteed, pre-emptible.
[22] Colab review summaries: free T4/K80, 12-hr runtime, ~12 GB RAM, disconnect on inactivity.

**Other free options worth one test, not a dependency:** Lightning AI free credits (~15 credits/mo in 2026 roundups), SageMaker Studio Lab (T4, 4 hrs/day, persistent 15 GB) [18][22] — verify eligibility from India / your account before counting them.

---

## 3. Data Cleaning Pipeline — Steps and Free Tools

Run cheap filters first. Every step below is open-source and runnable on Kaggle CPU (unlimited-ish) or Termux on samples, so GPU quota is never burned on data prep.

```
Raw (streamed) → Licence filter → Language / file filter → Heuristic quality
→ Exact dedup → Near-dedup (MinHash) → PII / secrets redaction
→ Decontamination vs benchmarks → Educational / complexity scoring
→ Tokenisation + packing → Shuffle (seeded) → Train / val split by repo
```

| Step | What to do (code-specific) | Free / OSS tools | FORGE default |
|---|---|---|---|
| **1. Licence filtering** | Keep only files where `license_type == permissive` AND SPDX in Blue Oak / ScanCode permissive list (MIT, Apache-2.0, BSD-2/3, ISC, etc.). Drop `no_license` for the redistributed mix — Stack v2 includes `no_license` files [2]; FORGE's clean release should not. Keep provenance (`repo_name, path, detected_licenses`) per file for attribution. | ScanCode Toolkit (if re-scanning), `license_stats.csv` from Stack v2 card, SPDX list | Stack-Edu fields already carry `detected_licenses` + `license_type` [5] — filter in `datasets` with `.filter()`, no re-scan needed at v0. |
| **2. Language & file filter** | Drop binary, vendor, generated (`is_vendor`, `is_generated`), minified JS, files >10 MB (Stack's own cut [2]), empty / >90% non-code lines. Keep `.py` (+ `.md` docstrings) at v0. | `go-enry` / linguist logic already applied in Stack; `datasets` filters | Python only at v0. Exclude `__pycache__`, `*.min.js` equivalents, lockfiles, data files masquerading as code. |
| **3. Heuristic quality (Gopher/C4-style for code)** | Remove: avg line length >100 chars, max line >1000, <10% alphanumeric, excessive repetition, auto-generated headers ("generated by", "do not edit"), low comment/docstring handling per task. StarCoder pipeline removed XML, filtered HTML boilerplate, language-specific filters after inspecting 30K files/language [15]. | DataTrove `C4QualityFilter`, custom Python filters, `regex` | Copy Stack / SantaCoder heuristics: SantaCoder training data (268 GB Python/Java/JS from Stack v1.1) was filtered on line length and % alphanumeric after dedup + PII [23]. Start there, ablate. |
| **4. Exact deduplication** | SHA-256 of normalised content (strip trailing whitespace). Remove exact duplicates within corpus. ~40% of permissively-licensed Stack v2 files were (near-)duplicates [2]. | Python `hashlib`, DataTrove exact dedup, `datasketch` | Mandatory. Cheapest win. |
| **5. Near-deduplication** | MinHash + LSH on 5-grams, **Jaccard ≥0.7** (StarCoder [15]) to **0.8** range — pick 0.75-0.8 for FORGE and ablate. Cluster, keep one representative (prefer higher Edu score, longer docstring, permissive licence). Stack paper: near-dedup gave **up to +33% relative HumanEval pass@1** vs raw in their ablations [9]. | **DataTrove** `MinhashDedupSignature → Buckets → Cluster → Filter` (n_grams=5, 14 buckets × 8 hashes default ≈0.72 threshold in docs) [24]; `datasketch MinHashLSH(threshold=0.8, num_perm=128)` for small samples [25] | Use DataTrove on Kaggle CPU for the 1-5B token slice; use datasketch on Termux only for prototyping the logic on 10K files. **Warning from 2026 v3 post-mortem coverage:** a MinHash shingle tokenisation bug in v2 over-deduplicated — validate your dedup on a hand-labelled duplicate set before trusting it [secondary]. |
| **6. PII & secrets redaction** | Regex + detector for emails, IPs, phone numbers, AWS/GitHub keys, `password=` patterns. FineWeb anonymised emails + public IPs [11]. The Stack may contain emails, IPs, API/SSH keys published to GitHub — card warns researchers to handle accordingly [2]. BigCode's StarPii is a StarEncoder-based PII detector from the project [26]. | `presidio-analyzer`, regex rules in DataTrove / NeMo Curator PII modifier, StarPii model (HF) for a second pass on samples | Regex pass is mandatory before any redistribution. Replace with `<EMAIL>`, `<IP>` placeholders, do not just drop the file (preserves code structure). Log redaction counts. Never commit a raw sample with a real key to GitHub. |
| **7. Educational / quality scoring** | Score files 0-5 for educational value (clear, self-contained, commented, algorithmic). Keep ≥3 (Stack-Edu recipe [3]). | Stack-Edu classifiers `HuggingFaceTB/stack-edu-classifier-python` etc. — StarEncoder + regression head trained on 500K Llama-3.1-annotated pairs [3]; or rule-based proxy at v0 (see §4) | If using Stack-Edu pre-filtered, this is already done. If using Python-Edu / raw Stack slice, run the published classifier on Kaggle GPU/CPU in batches — do not train your own classifier at v0. |
| **8. Tokenisation & packing** | Train BPE tokeniser on code (see architecture brief — not this doc), pack to fixed context (1024-2048 for v0) with proper EOS, **do not split a function arbitrarily if repo packing is available** (Stack train sets group by repo [2]). | HF `tokenizers`, DataTrove `DocumentTokenizer` | Pre-tokenise once, save as parquet / `.bin` shards ~100 MB each (nanochat pattern [12]), upload to HF as a private-then-public dataset. Training then streams shards, not raw text. |

[23] SantaCoder paper (arXiv:2301.03988) §5.1: 268 GB subset, near-dedup, PII redaction, line-length / alphanumeric filtering, decontamination (see §4).
[24] DataTrove DeepWiki / repo: Minhash system classes and defaults.
[25] Scalable Book §19.4: datasketch 5-line MinHashLSH example and DataTrove equivalence, dedup retention example.
[26] BigCode README summary: StarPii listed as StarEncoder-based PII detector.

**Pipeline implementation rule:** Every filter logs `in → out` counts and drops a sampled `rejected.jsonl` (100 examples) for manual inspection. If you cannot explain why a filter helps on a 350M ablation, remove it — complexity without ablation evidence is debt.

---

## 4. Contamination Avoidance Plan

Contamination is FORGE's credibility risk. The Stack itself is known to overlap benchmarks: the BigCode dedup write-up notes they **had to retrain because HumanEval was published in a GitHub repo in Python**, and that MBPP shares near-identical problems with LeetCode solutions abundant on GitHub [27]. Surveys confirm HumanEval/MBPP are contaminated in The Pile and especially The Stack, inflating CodeGen-class models [28].

### Protected benchmarks (never train on, never tune on final test)

HumanEval (164 problems), MBPP (974 problems, use sanitised / full splits consistently), MultiPL-E translations, APPS, plus FORGE's own held-out set (see below). Keep a **frozen final test** that is only run at milestones, not during weekly tuning.

### Three-layer decontamination (all layers, not one)

1. **Exact / n-gram overlap (cheap, mandatory)**
   - Normalise both sides: lowercase, strip punctuation / whitespace (lm-eval Janitor method), form **10-grams** (Cosmopedia decontamination uses 10-grams + lookup) [29] and also 8-gram code-token version.
   - Build a lookup of all benchmark n-grams (prompts + canonical solutions + test cases). Drop any training file containing a match. lm-eval's `decontaminate.py` implements `{ngram: doc_ids}` → scan → cache overlaps [29].
   - SantaCoder precedent: "decontaminated by removing files that contained test samples from HumanEval, APPS, MBPP and MultiPL-E" [23] — do exactly this list.

2. **Near-duplicate vs benchmarks (MinHash)**
   - MinHash every training file and every benchmark solution; drop training files with **Jaccard ≥0.7** to any benchmark item (same threshold family as StarCoder dedup [15]). This catches renamed-variable copies that n-grams miss.
   - Also do repo-level check: if a repo is named `humaneval-solutions`, `mbpp-answers`, `leetcode-python` with benchmark-identical problems, drop the repo wholesale and allowlist-review it manually.

3. **Semantic / embedding check (sampled, because n-grams are weak for code)**
   - Phi-1's analysis found **n-gram overlap fails in the coding domain** — it cannot capture logic similarity — while embedding retrieval (CodeGen-Mono 350M embeddings, L2 distance) was effective for CodeExercises vs HumanEval [28].
   - FORGE cheap version: embed benchmark problems + a random 50K training sample with a small code encoder (StarEncoder / `sentence-transformers` code model) on Kaggle GPU, flag cosine ≥0.85 for manual review. Run this once per dataset version, not per epoch.

### Process rules

- **Split by repository, never by file.** Stack card warns: "dataset contains a lot of near-duplicates which can cause leakage into the test split" if you split naively [30]. FORGE val set = whole held-out repos.
- **FORGE held-out set:** Create 200-300 fresh problems written / transformed after the training-data freeze date, with tests, stored **outside** the training repo (private HF dataset). Never upload them to public GitHub where they can be scraped into a future Stack.
- **Decontamination report per release:** Publish counts removed per benchmark per layer, the thresholds used, and 20 example (redacted) flagged files. A score without a decontamination report is not reported.
- **Rephrased-sample warning:** Research shows rephrased benchmark items evade n-gram filters and can inflate a 13B model to near-GPT-4 scores in controlled tests [31]. This is why layer 3 exists and why FORGE must not generate synthetic data by paraphrasing HumanEval/MBPP prompts (see §5).

[27] BigCode / foundation-model-stack dedup blog: HumanEval-in-GitHub retraining anecdote, MBPP ≃ LeetCode overlap.
[28] Contamination survey arXiv:2404.00699: embedding vs n-gram for code, Phi-1 CodeExercises analysis, Pile/Stack contamination of HumanEval/MBPP.
[29] lm-eval decontamination docs / Cosmopedia decontamination DeepWiki: Janitor normalisation, 10-gram lookup, SequenceMatcher verification with ratio threshold 0.5.
[30] HF `bigcode/the-stack` card via summary: no splits, near-duplicate leakage warning.
[31] EmergentMind summary of rephrased-samples paper arXiv:2311.04850: LLM-based decontaminator proposal, RedPajama / CodeAlpaca contamination findings.

---

## 5. Synthetic Code — Generation and Filtering

Synthetic data is FORGE's lever to beat raw-code-only training at small scale — **if** it is execution-filtered. Unfiltered synthetic code teaches confident wrong answers.

### What works in the literature (to copy at small scale)

- **OSS-Instruct (Magicoder):** Seed a teacher with a real open-source snippet, ask it to invent a *new* problem + solution inspired by (not copying) the snippet. Mitigates the bias of purely self-instruct data [13]. 75K set above.
- **Evol-Instruct:** Progressively complicate a seed problem (add constraints, edge cases). Useful for curriculum (§6) but easier to generate unsolvable problems — filter harder.
- **Genetic-Instruct style (recent work):** Crossover (merge constraints from 5 seeds) + mutation (perturb a task), then generate 3 candidate solution-test pairs, **execute in a secure container, keep the first passing one**, judge-LLM final check, dedup via MiniLM embeddings + FAISS cosine ≥0.90 + union-find clustering, and leakage check vs HumanEval/MBPP reporting zero overlap in that study [32]. FORGE cannot run a 27B judge, but can copy the *execution + dedup* skeleton.

[32] Moonlight review of "Increasing LLM Coding Capabilities through Diverse Synthetic Coding Tasks": Genetic-Instruct pipeline, Apptainer execution, FAISS dedup, HumanEval/MBPP leakage check.

### FORGE synthetic pipeline (₹0 version)

```
Seed (Stack-Edu file / textbook exercise, NOT a benchmark problem)
  → Teacher generates problem + solution + tests
    → STATIC: parses (ast.parse), no banned imports, timeout-bounded
    → EXECUTION: run tests in isolated subprocess (Kaggle/Colab: python -m pytest
      with 5s timeout, no network, ulimit; Termux: same for samples only)
    → Keep ONLY if: all teacher tests pass AND solution passes a second,
      independently generated test set (or property-based tests via hypothesis)
    → Dedup vs train set + vs benchmarks (§4 layers 1-2)
    → Complexity tag (§6) → SFT / annealing mix
```

- **Teacher problem on ₹0:** You have no paid API budget. Options, in order: (a) use the existing Magicoder 75K/110K sets instead of generating your own at v0; (b) use a free-tier open model on HF Inference / Colab (e.g., a 7-14B coder running locally in the notebook) as teacher for a **few thousand** examples, accepting lower quality; (c) self-improvement loop **only after** FORGE v0 exists — sample k=10 solutions per training problem at temp 0.8, keep those that pass tests (rejection sampling / RFT), retrain. Never present (c) as available before v0 exists.
- **Execution is the filter, not the teacher's confidence.** Report the pass rate: if only 30% of generated solutions pass their own tests, that is the expected reality — publish it, do not lower the bar to inflate dataset size.
- **Unit-test-driven selection for real code too:** For Stack-Edu files that are part of repos with tests, prefer files whose tests are present. At v0 this is a *sampling preference* (upweight repos with `test_*.py`), not a full execution of every repo (too expensive).
- **Hard bans:** No synthetic problem that is a paraphrase of HumanEval/MBPP/APPS (check §4), no solution without tests, no "explanation-only" SFT data counted as code data.

---

## 6. Curriculum Learning for Code

Evidence for strict easy→hard ordering in *pretraining* is mixed; evidence for **annealing** (ending training on the highest-quality / most task-relevant data) and for **quality filtering** is stronger (Stack-Edu gains [3], OpenCoder annealing corpus [8], SmolLM2 annealing on 200B tokens in ablations [3]). FORGE should implement curriculum as *phases*, not as a fragile per-example sort:

| Phase | Data | Purpose | Share of FORGE v1 budget |
|---|---|---|---|
| **P1 — Foundations** | Stack-Edu Python, short files (<2K tokens), high Edu score, single-function / single-class, heavily commented | Syntax, idioms, docstring→code mapping | First 40% of steps |
| **P2 — General code** | Full Stack-Edu Python mix, all lengths, multi-function files, repo-grouped context | Generalisation, longer dependencies | Next 40% of steps |
| **P3 — Annealing** | Highest Edu score slice + algorithmic problems (OpenCoder-annealing style) + execution-filtered synthetic (§5) + textbook-style exercises (Python-Edu top slice) | Sharpen problem-solving before eval; SmolLM2-style linear LR decay during annealing | Final 20% of steps, LR decaying to ~0 |

**Complexity proxies (free, computable on CPU):** file length in tokens, AST depth / number of functions & classes (`ast` module), cyclomatic complexity (`radon` / `lizard` — pip, free), Edu classifier score, # external imports. Tag every pre-tokenised shard with median complexity so phases are just shard selections.

**What not to do:** Do not sort the *entire* training set by complexity and train in a single monotonic pass — the model forgets early patterns and you lose the benefit of shuffling. Phase-level curriculum with shuffling *within* each phase is the defensible default. Ablate it: P1→P2→P3 vs shuffled-all — if the ablation shows no gain on held-out pass@1, drop the curriculum claim.

---

## 7. Realistic Data Budget for Free Compute

### The maths that sets the budget

- **Chinchilla rule of thumb:** ~**20 tokens per parameter** is compute-optimal (DeepMind: model size and tokens scale in equal proportion, `C ≈ 6·N·D`) [33]. Modern practice *over-trains* small models past Chinchilla because inference cost matters (LLaMA-3 70B at 214 tokens/param in summaries [33]) — for FORGE, over-training a 125-350M model is desirable **if** you have the GPU weeks, but Chinchilla is the floor for a credible run.
- **FLOPs estimate:** `C ≈ 6·N·D`. On a T4 (16 GB, ~65 TFLOPS FP16 peak, realistic 25-35% MFU for a small custom trainer ≈ 16-22 TFLOPS effective), wall-clock ≈ `C / effective_FLOPS`.

[33] Chinchilla summaries: Hoffmann et al. 2022, 20 tokens/param, `C ≈ 6ND`, Gopher 280B/300B vs Chinchilla 70B/1.4T comparison.

### Worked budgets (single T4, bf16/fp16, seq 1024-2048)

| Target | Tokens (D) | FLOPs ≈6ND | Est. T4 time @ ~18 TFLOPS eff. | Kaggle weeks @30 hrs/wk | Verdict |
|---|---|---|---|---|---|
| 125M, prototype | 0.5B | 3.8×10¹⁷ | ~6 hrs | <1 week inc. failures | **Week-1 smoke test.** Proves pipeline, tokeniser, checkpoint/resume, eval harness. Expect near-zero HumanEval — that is fine. |
| 125M, Chinchilla | 2.5B | 1.9×10¹⁸ | ~29 hrs | ~1-1.5 weeks | **Credible v0.** First model where held-out pass@1 is meaningful. |
| 350M, pragmatic | 2.0B | 4.2×10¹⁸ | ~65 hrs | ~2-3 weeks | Under-trained but a good ablation vs 125M/2.5B at similar compute. |
| 350M, Chinchilla | 7.0B | 1.5×10¹⁹ | ~228 hrs | ~7-8 weeks Kaggle alone; ~4-5 weeks Kaggle+Colab opportunistic | **FORGE v1 target.** Only attempt after v0 eval + dedup/decontamination are proven. Requires checkpoint discipline across ~30+ sessions. |
| 350M, over-trained | 10B | 2.1×10¹⁹ | ~325 hrs | ~10+ weeks | Stretch goal. Diminishing returns vs spending the same time on SFT + execution-filtered synthetic + eval. |

*Estimates are order-of-magnitude, not promises — MFU depends entirely on your trainer (batch size, gradient checkpointing, fused kernels). Measure tokens/sec in the prototype run and recompute this table with your real number before committing to v1.*

### Storage / bandwidth budget (the hidden constraint)

- 1B tokens of code ≈ 3-4 GB as raw text, ~4-5 GB tokenised as `uint16/uint32` shards. **2.5B tokens ≈ 10-12 GB** — fits Kaggle's 20 GB persistent disk *barely*, with checkpoints. **7B tokens ≈ 28-35 GB — does not fit in one Kaggle working dir.** Solution: keep shards on HF Hub dataset repo, stream / download per-phase shards, delete after use.
- On the phone: never store >2 GB of corpus locally. Stream everything.
- Checkpoint size: 350M in bf16 ≈ 700 MB (+ optimiser state 2-3× if saved — **save optimiser only on Kaggle working disk for resume within the week, push weights-only to HF**).

### Recommended FORGE data budget (decision)

- **v0 (prove it):** 2-3B tokens, Python-Edu / Stack-Edu Python, score ≥3, deduped + decontaminated. Single language.
- **v1 (if v0 beats a memorisation baseline):** 6-8B tokens, 85% code (Stack-Edu Python + 1-2 extra languages) + 10% FineWeb-Edu sample + 5% annealing/synthetic by token count in P3.
- **SFT:** 50-100K execution-filtered examples (Magicoder subset + rejection-sampled), 2 epochs, counted separately from pretraining tokens.
- **Validation:** 5-10M tokens held out **by repo** + frozen benchmark suite. Never train on it, even "just a little at the end."

---

## 8. Risks and Open Questions

1. **Stack-Edu / Python-Edu access path** — confirm the exact HF repo ids and gating in a live `huggingface_hub` listing before writing download code; one path probed here returned 401. Fallback (Stack-Edu Python slice) must be tested in the prototype week.
2. **Licence of synthetic teacher data** — Magicoder sets carry MIT/Apache labels on HF, but were generated with `gpt-3.5-turbo`. Get a clear position in the FORGE README: what is redistributed (weights? data? both?) and under what terms.
3. **`no_license` files** — Stack v2's inclusion of `no_license` ≠ permission to redistribute. FORGE clean mix = permissive-only. Document the drop rate.
4. **Free-tier volatility** — Kaggle/Colab quotas are not contracts. The roadmap must survive a 50% quota cut: that is another reason v0 is 125M, not 1B.
5. **Dedup bug risk** — validate MinHash on synthetic duplicate pairs (exact copy, renamed vars, reformatted) and report precision/recall before the full run.

---

## Sources (primary first)

- Hugging Face dataset cards: `bigcode/the-stack-v2`, `bigcode/the-stack-v2-dedup`, `HuggingFaceTB/stack-edu`, `HuggingFaceTB/stack-edu-classifier-*`, `HuggingFaceFW/fineweb`, `HuggingFaceTB/SmolLM-135M/360M`, `HuggingFaceTB/SmolLM2-360M-Instruct`, `infly/OpenCoder-1.5B-Base`, `infly/OpenCoder-8B-Instruct`, `ise-uiuc/Magicoder-*`
- Papers: The Stack (arXiv:2211.15533), StarCoder (arXiv:2305.06161), SantaCoder (arXiv:2301.03988), FineWeb (arXiv:2406.17557), Magicoder (arXiv:2312.02120), OpenCoder (arXiv:2411.04905), Chinchilla (Hoffmann et al. 2022), contamination survey (arXiv:2404.00699), rephrased samples (arXiv:2311.04850)
- Code / docs: BigCode GitHub + dedup blog, DataTrove repo + DeepWiki, lm-evaluation-harness decontamination docs, OpenCoder-llm repo README (via DeepWiki), Kaggle `docs/efficient-gpu-usage` (via AgentDeals listing read 2026-10-23)
- Free-tier roundups (secondary, cross-checked across ≥2 sources): Spheron 2026, ThunderCompute Colab tiers, AIKetra Kaggle Review 2026, AgentDeals Kaggle listing

*Prepared for FORGE — dataset engineering track. Next action: prototype-week script that streams 10K Stack-Edu Python files, runs §3 steps 1-6, and reports in/out counts + dedup validation — before any GPU training is booked.*
