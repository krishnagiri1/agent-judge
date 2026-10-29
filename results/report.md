# Results

- Transcripts judged: **500** public SWE-agent runs on **250** real GitHub issues (`nebius/SWE-agent-trajectories`, CC-BY-4.0), balanced 50/50 between runs that passed and failed the repository's tests.
- Human labels: **50** transcripts hand-scored with the same rubric and the same rendered text the judge sees (blind to the test outcome and the agent model).

## Predicting the hidden test outcome (primary result)

AUROC of each rater's `resolved_probability` and `correctness` against whether the patch actually passed the repository's tests (0.5 = chance, 1.0 = perfect), with a 95% bootstrap CI.

| Rater | Prompt | n (all) | AUROC P(resolved) | AUROC correctness | Parse errors | n (labeled) | AUROC P(resolved), labeled subset |
|---|---|---|---|---|---|---|---|
| Heuristic baseline (no LLM) | v2 | 500 | 0.65 [0.60, 0.70] | 0.65 | 0% | 50 | 0.68 |
| ollama:qwen2.5:14b | v1 | 500 | 0.71 [0.66, 0.75] | 0.71 | 0% | 50 | 0.61 |
| ollama:qwen2.5:14b | v2 | 500 | 0.67 [0.62, 0.72] | 0.67 | 0% | 50 | 0.58 |
| **Human (you)** | – | – | – | – | – | 50 | 0.92 |

## Judge vs. human agreement

Quadratic-weighted Cohen's kappa on the 1-5 scale (95% bootstrap CI for overall).

| Judge | Prompt | n | κ correctness | κ tool_use | κ reasoning | κ overall | Spearman (overall) | Within ±1 (overall) | Parse errors |
|---|---|---|---|---|---|---|---|---|---|
| Heuristic baseline (no LLM) | v2 | 50 | 0.22 | 0.27 | 0.12 | 0.19 [0.08, 0.33] | 0.45 | 52% | 0% |
| ollama:qwen2.5:14b | v1 | 50 | 0.22 | 0.02 | 0.12 | 0.13 [0.04, 0.22] | 0.43 | 38% | 0% |
| ollama:qwen2.5:14b | v2 | 50 | 0.18 | 0.06 | 0.13 | 0.13 [0.05, 0.21] | 0.43 | 32% | 0% |

## Position bias (pairwise, both orders)

Each pair is one passing and one failing run on the same issue, judged twice with the order swapped.

| Judge | Prompt | Pairs | Verdict flips with order | 'A' win rate | Accuracy (single order) | Accuracy (swap-aggregated) |
|---|---|---|---|---|---|---|
| ollama:qwen2.5:14b | baseline | 100 | 30% | 57% | 71% | 71% |
| ollama:qwen2.5:14b | debiased | 100 | 41% | 40% | 68% | 66% |

**ollama:qwen2.5:14b:** the debiased prompt changed the order-flip rate from 30% to 41% (37% relative increase).

## Verbosity bias (content-free narration added to every agent turn)

| Judge | Prompt | n | Δ overall | Δ reasoning | Δ correctness | Δ P(resolved) | Share scored higher (overall) | Wilcoxon p (overall) |
|---|---|---|---|---|---|---|---|---|
| ollama:qwen2.5:14b | v2 | 100 | 0.00 | -0.04 | -0.04 | -0.3 | 11% | 1.000 |
