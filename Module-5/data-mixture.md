# Data Mixture & Curriculum Plan (100B Token Budget Baseline)


## 0. Target capabilities → benchmarks → data shape

| Capability | Benchmark(s) | Training data shape | Loss mask |
|---|---|---|---|
| Coding / code-editing | SWE-bench / HumanEval / MBPP | repo + bug report → patch | loss on patch only |
| Tool use / agentic | **AgentBench / BFCL (Berkeley Function-Calling Leaderboard), held-out tool-call-format-validity set for the §8 proxy metric** | multi-step trajectory: plan → tool call → observation → recover → answer | loss on plan/calls/answer; **no loss on observations** |
| Reasoning (controllable effort) | GSM8K / MATH / SVAMP, low/med/high/ultra effort bands | worked traces of varying length | loss on trace + answer |
| Indic (native) | **IndicXTREME (classification/QA) + Flores-200 Indic pairs (translation) + MILU (multitask understanding)** | native-language text, QA, translation | loss on target language tokens |
| General knowledge/world model | MMLU-style / ARC / HellaSwag | general web/textbook text | standard LM loss |

*Primary benchmark selected per row for §8 proxy evaluation.*

## 1. Main pretraining mixture (token-budget allocation)

*Based on a Total Budget of **100 Billion Tokens (100,000,000,000 tokens)**.*

| Lane | Share | Tokens (of 100B Total Budget) | Real supply status & Pipeline Source |
|---|---|---|---|
| General web | 42.0% | **42.0 Billion tokens** (42,000,000,000) | Abundant: 4 sub-pools (§1a), 61% of pipeline yield (Wiki 12.39B, Web 10.33B, Lit 9.66B, Comm 9.66B) |
| Code | 23.0% | **23.0 Billion tokens** (23,000,000,000) | Code&Math seed pool (16% of run) expanded via dedicated StarCoder2 / The Stack v2 pipeline |
| Math / science / reasoning | 15.0% | **15.0 Billion tokens** (15,000,000,000) | D1 Science (arXiv, 23% of run) + OpenWebMath usable as raw pretraining text |
| Indic (protected floor) | 10.0% | **10.0 Billion tokens** (10,000,000,000) | Stage 1 reconfigured pipeline run on AI4Bharat Sangraha + IndicCorp v2 |
| Agentic (protected floor) | 5.0% | **5.0 Billion tokens** (5,000,000,000) | Sourced (ToolBench/Gorilla 1.5B) + Rejection-sampled synthetic trajectories (3.5B) |
| Long-context | 5.0% | **5.0 Billion tokens** (5,000,000,000) | Books, long technical preprints & synthetic long-context documents (§7) |
| **Total** | **100.0%** | **100.0 Billion tokens** | **Fully Allocated** |

## 1a. Benchmark Inventory → Lane Mapping (Cleaned Data Baseline)

*(From 100MB benchmark pipeline run: 154.2 MB input → 100.0 MB clean yield, ~27,800 clean docs)*

| Domain pool | Docs (of 27,800) | Share of sample | Target Lane | Notes |
|---|---|---|---|---|
| D1 Science (arXiv) | 6,394 | 23.0% | Math/science/reasoning | usable as-is |
| D1 Knowledge (Wikipedia) | 5,004 | 18.0% | General web — educational/bridge sub-tier | usable as-is |
| D2 Reasoning (Code & Math) | 4,448 | 16.0% | Code lane | expanded with StarCoder2 |
| D2 Literature (Book & Doc) | 3,892 | 14.0% | General web — quality/structured sub-tier | usable as-is |
| D3 General Web (Web Page) | 4,170 | 15.0% | General web — diversity sub-tier | usable as-is |
| D4 Community (News/Forum) | 3,892 | 14.0% | General web — noisy-diversity sub-tier | usable as-is |
| Indic (any tier) | 0 | 0% | Indic lane | absent by construction (reconfigured filter) |
| Agentic trajectories | 0 | 0% | Agentic lane | absent by construction (synthetic generation) |

### Internal general-web sub-mix, normalized

The four general-web sub-pools sum to 61% of the baseline yield (17,958 docs out of 27,800). Renormalized to 100% *within the 42.0B token General Web lane*:

| Sub-pool | Share of general-web lane | Tokens in 42.0B General Web Lane |
|---|---|---|
| D1 Knowledge (Wikipedia) — educational bridge | 18/61 = **29.5%** | **12.39 Billion tokens** |
| D3 General Web — diverse filtered | 15/61 = **24.6%** | **10.33 Billion tokens** |
| D2 Literature — quality/structured | 14/61 = **23.0%** | **9.66 Billion tokens** |
| D4 Community — noisy diversity | 14/61 = **23.0%** | **9.66 Billion tokens** |
| **General Web Total** | **100.0%** | **42.00 Billion tokens** |

## 2. Indic tier split (protected lane: 10.0B tokens)

*Real inventory breakdown for the 10.0 Billion token Indic budget:*

| Tier | Share of Indic budget | Tokens (of 10B Indic) | Target Datasets / Sources | Notes |
|---|---|---|---|---|
| Verified native | 55.0% | **5.50 Billion tokens** | AI4Bharat Sangraha, IndicCorp v2 | Highest trust; sorted by verified native tokens |
| Unverified native/scraped | 20.0% | **2.00 Billion tokens** | mC4 Indic, OSCAR Indic | Broadens vocabulary & dialectal diversity |
| Translated | 15.0% | **1.50 Billion tokens** | BPCC, Flores-200 Indic parallel pairs | Capped to avoid translationese artifacts |
| Synthetic | 10.0% | **1.00 Billion tokens** | Indic-Instruct, synthetic QA pairs | High educational score, strict synthetic cap |
| **Indic Total** | **100.0%** | **10.00 Billion tokens** | | **Allocated across 22 scheduled languages** |

## 3. Agentic slot (protected lane: 5.0B tokens)

- **Total Lane Volume:** **5.00 Billion tokens** (~2.5 million multi-step trajectories at ~2,000 avg tokens/trajectory).
- **Source Breakdown:**
  - **Sourced Open Datasets (30% = 1.50B tokens):**
    - ToolBench trajectories: **0.60B tokens** (~300k trajectories)
    - Gorilla / APIBench: **0.40B tokens** (~200k trajectories)
    - AgentInstruct: **0.30B tokens** (~150k trajectories)
    - WebArena trajectories: **0.20B tokens** (~100k trajectories)
  - **Synthesized Trajectories (70% = 3.50B tokens):**
    - Rejection-sampled synthetic trajectories (~1.75M trajectories) generated via self-executing API sandbox with strict format validity checks.
- Most of this lane is synthesized (real multi-step trajectories are absent from standard web curation and require dedicated generation).
- **Annealing Reserve:** **0.75B tokens** (15% of top Tier A agentic trajectories) held back for §5 final annealing phase.

## 4. Reasoning slot (effort bands: 15.0B tokens)

*Allocated across effort bands for the 15.0 Billion token reasoning budget:*

| Band | Share of reasoning budget | Tokens (of 15B Reasoning) | Domains | Approx. trace length |
|---|---|---|---|---|
| Low effort | 40.0% | **6.00 Billion tokens** | math, code, general | short, direct (<256 tokens) |
| Medium effort | 35.0% | **5.25 Billion tokens** | math, code, general | multi-step, linear (256–1024 tokens) |
| High effort | 15.0% | **2.25 Billion tokens** | math, code | step-by-step verification (1024–4096 tokens) |
| Ultra effort | 10.0% | **1.50 Billion tokens** | math, code | search tree & self-correction (>4096 tokens) |
| **Reasoning Total** | **100.0%** | **15.00 Billion tokens** | | |

*Spent across pretraining and fine-tuning (SFT + RLVR).*

## 5. Protected floors, selector, and annealing reserve

- **Always-on floor:** Indic ≥ **8.0B tokens** (8%), Agentic ≥ **3.0B tokens** (3%), Reasoning ≥ **10.0B tokens** (10%) of every OPUS-selected batch, regardless of selector score.
- **Anneal reserve (15% held back):** Total of **2.1375 Billion tokens** (~2.14B tokens) held out of main run for final annealing phase:
  - **0.825B tokens** (15% of 5.5B verified Indic native)
  - **0.750B tokens** (15% of 5.0B agentic trajectories)
  - **0.5625B tokens** (15% of 3.75B high + ultra reasoning traces)

*Spent strictly during the final 2% of training steps to prevent early overfitting while maximizing final capability metrics.*

## 6. Design note: why (and where) the 50-30-20 quality-mix idea is used

*(See §1a: baseline starts at 29.5/24.6/23.0/23.0, and §8 Phase 1 proxy sweep searches toward 50-30-20)*

## 7. Curriculum (stage-by-stage)

| Stage | General web (42B) | Code (23B) | Reasoning (15B) | Indic (10B) | Agentic (5B) | Long-context (5B) |
|---|---|---|---|---|---|---|
| Seed / early (0–10%) | 20.0B | 1.0B | 0.5B | 0.8B (8% floor) | 0.3B (3% floor) | 0.0B |
| Mid (10–70%) | 18.0B | 14.0B | 8.0B | 5.2B | 2.45B | 2.5B |
| Late / pre-anneal (70–98%) | 3.5B | 6.5B | 5.5B | 3.175B | 1.5B | 2.5B |
| Anneal (98–100%) | 0.5B | 1.5B | 1.0B | 0.825B (Reserve) | 0.750B (Reserve) | maintained |
| **Total** | **42.0B** | **23.0B** | **15.0B** | **10.0B** | **5.0B** | **5.0B** |

- **Warmup band:** **4.0% of total budget (4.0 Billion tokens)**, covering the first 40,000 steps at batch size 100,000 tokens. Eliminates hard cutoffs and prevents the ~150× gradient-norm spikes observed in V4.

## 8. Proxy experiment plan

*Phase 1 starts immediately at 20M–25M token proxy scale using the benchmark split baseline.*

## 9. Metrics & Open Risks Summary

- **Collection Pipeline Fact:** Standard web runs yield 0% Indic & Agentic tokens by construction; resolved via parallel Indic pipeline (10B tokens) and synthetic/sourced agentic generation (5B tokens).
- **Total Token Budget:** Set to **100 Billion tokens (100B)**, unblocking all absolute token counts in §1, §2, §3, §4, §5, and §7.
- **Indic Verified-Native Supply:** Budgeted at **5.5B verified native tokens** (AI4Bharat Sangraha + IndicCorp v2) out of 10B Indic total.
- **Agentic Supply & Licensing:** Budgeted at **1.5B open sourced tokens** (ToolBench, Gorilla, AgentInstruct, WebArena) + **3.5B rejection-sampled synthetic tokens**.
- **Anneal Reserve:** Locked at **2.14B tokens** (15% of verified Indic, agentic, and high/ultra reasoning) reserved for final 2% annealing phase.
- **Code Lane:** Expanded to **23.0B tokens** (23%) using dedicated StarCoder2 / The Stack v2 corpus with AST-level deduplication and compilability verification.


