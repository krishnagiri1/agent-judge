# LLM-as-Judge for Coding-Agent Transcripts

How far can you trust an LLM to grade an AI coding agent? This project builds a rubric-based LLM judge
for SWE-agent transcripts and measures it against the one thing that can't be argued with: whether each
run's patch **actually passed the repository's tests**. A smaller hand-labeled set checks agreement with a
human reviewer, and two experiments stress-test the judge for well-known failure modes: **position bias**
and **verbosity bias**.

- **Data:** public SWE-agent runs on real GitHub issues
  ([`nebius/SWE-agent-trajectories`](https://huggingface.co/datasets/nebius/SWE-agent-trajectories), CC-BY-4.0).
  Every run comes with whether its patch actually passed the repository's tests, which gives an
  objective check that most LLM-judge projects lack.
- **Judge:** any OpenAI-compatible API, Anthropic, or a free local model through Ollama.
- **Metrics:** AUROC vs. test outcomes (primary, no labeling needed), quadratic-weighted Cohen's κ and
  Spearman vs. 50 human labels, order-flip rate under swapped pairs, and score shift under content-free padding.
- **Tracking:** every run logged to MLflow.
- **Demo:** [browse every verdict against the real outcome](https://krishnagiri1.github.io/agent-judge/) (static page), or judge a transcript live with the local Gradio app.

## Results

Judge: `qwen2.5:14b` running locally through Ollama on a laptop RTX 4080. Full tables in
[`results/report.md`](results/report.md); every run is also logged to MLflow.

### Key findings

1. **The 14B judge has real but modest skill, and barely beats a simple rule.** It predicts which
   patches pass the hidden tests with AUROC **0.71** [95% CI 0.66–0.75]. A no-LLM heuristic (did the
   agent submit? how many errors? how big is the patch?) reaches **0.65** [0.60–0.70]. On the 50
   transcripts I labeled by hand, my own estimates reached **0.92**.
2. **It is strongly lenient.** Its average overall score is **3.72/5** vs. **2.14** for the human
   labels (+1.6 points). That offset is why quadratic-weighted κ is only **0.13** even though rank
   agreement is moderate (Spearman **0.43**).
3. **It is position-biased.** When two attempts at the same issue are shown in swapped order, **30%**
   of verdicts flip, and the first transcript wins 57% of the time.
4. **It is not verbosity-biased.** Adding content-free narration to every agent turn changed scores by
   **0.00** on average (Wilcoxon p = 1.0).
5. **Both prompt "fixes" backfired.** A checklist-style prompt (v2) increased leniency (+2.1 points)
   and lowered AUROC to 0.67. A debiasing prompt raised the order-flip rate from **30% to 41%** and
   pushed the judge to answer "tie" 36% of the time.

### Predicting the hidden test outcome (500 transcripts, 50/50 pass/fail)

| Rater | Prompt | AUROC P(pass) [95% CI] | AUROC on the 50 labeled transcripts |
|---|---|---|---|
| Heuristic baseline (no LLM) | – | 0.65 [0.60, 0.70] | 0.68 |
| qwen2.5:14b | v1 (direct scoring) | **0.71** [0.66, 0.75] | 0.61 |
| qwen2.5:14b | v2 (checklist) | 0.67 [0.62, 0.72] | 0.58 |
| Human (me) | – | – | **0.92** |

### Agreement with human labels (50 transcripts)

| Rater | Prompt | κ overall [95% CI] | Spearman | Within ±1 point | Mean score (human = 2.14) |
|---|---|---|---|---|---|
| Heuristic baseline | – | 0.19 [0.08, 0.33] | 0.45 | 52% | – |
| qwen2.5:14b | v1 | 0.13 [0.04, 0.22] | 0.43 | 38% | 3.72 |
| qwen2.5:14b | v2 | 0.13 [0.05, 0.21] | 0.43 | 32% | 4.28 |

### Bias experiments

| Experiment | Baseline prompt | "Debiased" prompt |
|---|---|---|
| Verdict flips when order is swapped (100 pairs) | 30% | 41% |
| First transcript wins (50% = unbiased) | 57% | 40% |
| "Tie" answers | 1% | 36% |
| Picks the run that actually passed | 71% | 66% |

| Verbosity test (100 transcripts) | Δ overall | Δ correctness | Share scored higher | Wilcoxon p |
|---|---|---|---|---|
| Content-free narration added to every agent turn | 0.00 | −0.04 | 11% | 1.00 |

### What I'd try next

- A frontier-model judge on the same 500 runs, to see how much model scale closes the gap to the human.
- Calibrating scores (e.g., mapping the judge's score distribution onto the human one) to remove the
  leniency offset, then re-measuring κ.
- Averaging verdicts over both orders (and several samples) as a cheaper fix for position bias than
  prompt instructions.

## How it works

```
nebius/SWE-agent-trajectories ──► sample: issues with ≥1 passing and ≥1 failing run
                                   (balanced 50/50, natural pairs for bias tests)
                                        │
                                        ▼
                       render: bounded transcript (issue · steps · final patch)
                       identical text for the human and the judge; test outcome hidden
                         │                                  │
                         ▼                                  ▼
              human labels (Gradio app)          LLM judge (rubric, prompt v1 / v2)
                         │                                  │
                         └──────────────► metrics ◄─────────┘
                                κ / Spearman vs human · AUROC vs tests
                                position bias (A/B swap) · verbosity bias (padding)
```

**Rubric** (`agent_judge/rubric.py`): 1–5 scores with written anchors for *correctness* (does the patch
fix the root cause), *tool use* (efficient, no thrashing), *reasoning* (reproduce → diagnose → fix →
verify) and *overall*, plus a 0–100 probability that the patch passes the hidden tests.

**Prompts** (`agent_judge/judge.py`): `v1` scores directly; `v2` first works through five explicit
checks (root cause, reproduction, patch relevance, verification, loops) and is told that narration is
not evidence. Comparing them is one of the experiments.

**Rendering** (`agent_judge/render.py`): runs average 8k–15k tokens, so each agent turn and tool output
is capped (keeping head and tail, where the action and the error message live), and long runs keep the
first 8 and the last steps with an explicit `[... N steps omitted ...]` marker.

**Position bias** (`agent_judge/bias.py`): for each issue, one passing and one failing run are judged
pairwise in both orders. Reported: how often the verdict flips with order, how often "A" wins (should be
50%), and accuracy against the tests with and without swap-aggregation. The `debiased` prompt assesses
each transcript on its own before comparing.

**Verbosity bias**: each transcript is judged as-is and with content-free narration ("Let me think about
this carefully…") added to every agent turn. Actions, outputs, and the patch are untouched, and the same
steps are kept in both renders, so any score increase is pure verbosity bias (tested with Wilcoxon).

**Experiment tracking**: `report` logs each judge × prompt run and each bias experiment to MLflow
(experiment `agent-judge`) with its parameters and metrics, plus the report and chart as artifacts, so
runs can be compared side by side in the MLflow UI.

**Heuristic baseline**: `--provider mock` is a no-LLM rule-based judge (submitted? errors? patch size?).
It makes every command run instantly for testing and gives a floor that a real judge must beat.

## Setup

Python 3.10+ on Windows, macOS, or Linux.

```bash
git clone <your-repo-url> agent-judge && cd agent-judge
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # 18 tests, no network or API key needed
```

Pick at least one judge:

- **Local and free (recommended to start):** install [Ollama](https://ollama.com), then pull a model
  that fits your GPU, e.g. `ollama pull qwen2.5:14b` (≈9 GB, fits a 12 GB card) or `ollama pull llama3.1:8b`.
  Any newer model that fits works too. Use `--workers 1` for local models.
- **API:** set `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) and pass that provider's model name. For other
  OpenAI-compatible providers, also pass `--base-url`.

Every LLM response is cached in `.cache/`, so re-running a command never pays twice.

## Run the project, step by step

```bash
# 1. Sample 250 issues -> 500 transcripts (downloads ~450 MB once)            ~10 min
python -m agent_judge sample --n-instances 250 --shards 5

# 2. Hand-label 50 transcripts (see docs/labeling-guide.md)                   ~5-6 h
python -m agent_judge label --n 50

# 3. Judge all 500 with each judge and prompt you want to compare             hours, unattended
python -m agent_judge judge --provider ollama --model qwen2.5:14b --prompt v1 --workers 1
python -m agent_judge judge --provider ollama --model qwen2.5:14b --prompt v2 --workers 1
python -m agent_judge judge --provider openai --model <model-name> --prompt v2
python -m agent_judge judge --provider mock --model heuristic --prompt v2     # baseline

# 4. Position bias: baseline vs. debiased prompt on 100 pairs (200 calls each)
python -m agent_judge position-bias --provider ollama --model qwen2.5:14b --prompt baseline --workers 1
python -m agent_judge position-bias --provider ollama --model qwen2.5:14b --prompt debiased --workers 1

# 5. Verbosity bias on 100 transcripts (re-uses cached originals)
python -m agent_judge verbosity-bias --provider ollama --model qwen2.5:14b --prompt v2 --workers 1

# 6. Build results/report.md, results/metrics.json and the kappa chart,
#    and log every run to MLflow
python -m agent_judge report
mlflow ui          # then open http://127.0.0.1:5000 to compare runs

# 7. Try the demo locally
python -m agent_judge demo
```

Tip: run each experiment on `--limit 20` (judge) or `--n-pairs 10` first to check the outputs look sane
before the full run. Every command is resumable; interrupting and re-running continues where it stopped.

**Cost of an API run:** each transcript is roughly 7–8k input tokens, so 500 transcripts is about
4M input tokens per judge × prompt. Check your provider's current pricing; small models make this cheap,
and local Ollama runs are free.

## Publish the demo on GitHub Pages (free)

The demo is a static page: every verdict is pre-computed, so it needs no server, GPU, or API key.

```bash
python -m agent_judge export-site      # writes docs/index.html + docs/data.js
```

Open `docs/index.html` locally to preview it, commit and push, then on GitHub go to
**Settings → Pages → Build and deployment**, choose **Deploy from a branch**, branch `main`, folder `/docs`,
and save. The page appears at `https://<username>.github.io/agent-judge/` within a few minutes.

For live judging with any model, run the Gradio app locally: `python -m agent_judge demo`.

## Project layout

```
agent_judge/
  data.py           download, sample and store transcripts
  render.py         bounded, leak-free transcript rendering
  rubric.py         the shared rubric (judge prompt + labeling app)
  llm.py            OpenAI-compatible / Anthropic / Ollama / heuristic clients + SQLite cache
  judge.py          pointwise and pairwise judging, robust JSON parsing with retries
  metrics.py        weighted kappa, Spearman, AUROC, bootstrap CIs
  bias.py           position- and verbosity-bias experiments
  report.py         results/report.md, metrics.json, chart
  tracking.py       MLflow logging of runs and bias experiments
  labeling_app.py   blind hand-labeling UI
  demo_app.py       Gradio demo (saved verdicts + live judging)
  site.py           static verdict browser for GitHub Pages
  cli.py            python -m agent_judge <command>
app.py              Gradio entry point
examples/           8 bundled transcripts for the demo
docs/               labeling guide + the published static demo
tests/              unit + end-to-end tests (mock judge, no network)
```

## Limitations

- One human annotator and 50 labels, so κ and the human AUROC have wide confidence intervals and mix
  judge error with labeler noise. That's why the test outcome on all 500 runs is the primary metric.
- One local judge model (14B). Conclusions about leniency and position bias may not hold for larger models.
- The sample is balanced 50/50 between passing and failing runs; real agent pass rates are lower, so
  AUROC (threshold-free) is reported rather than accuracy.
- Transcripts are truncated to fit local-model context windows. Human and judge see the same text, but
  both may miss evidence in omitted steps.
- The agents in this dataset are 2024-era Llama-based SWE-agent runs; judges may behave differently on
  transcripts from other agent frameworks.

## Data and license

Code: MIT. Transcripts: `nebius/SWE-agent-trajectories` by Nebius, CC-BY-4.0. Each underlying repository
keeps its own license.
