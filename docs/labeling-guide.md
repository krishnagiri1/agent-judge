# Labeling guide

The test outcome of each run is the project's main ground truth. Your 50 labels answer a second question:
does the judge agree with a careful human reviewer? Consistency matters more than speed. Plan on 6–8
minutes per transcript; 50 transcripts is roughly 5–6 hours.

## How to read a transcript

1. **Read the issue first.** Write down (mentally or in Notes) what the root cause probably is.
2. **Jump to the final patch.** Does it touch the code responsible for that root cause? Does it leave
   behind debug prints, reproduction scripts, or unrelated edits?
3. **Skim the steps.** Look for: reproducing the bug, finding the right file, failing edits that
   repeat, ignored errors, and whether the agent re-ran the reproduction after the fix.
4. **Score each dimension against its anchors** (open the Rubric panel). Use the anchor text, not a
   gut feeling about the whole run. A clean process with a wrong fix can be tool_use 4, correctness 2.
5. **Estimate the probability the patch passes the hidden tests.** Use the full range: 10 or 90 when
   you're fairly sure, 50 only when you genuinely can't tell.

## Consistency habits

- Label your first 10, then re-read the rubric and fix any of those 10 you scored differently than
  the anchors say. Early labels drift the most.
- Don't label more than ~25 in one sitting; fatigue makes scores creep toward 3.
- When the transcript says `[... N steps omitted ...]`, judge only what is shown. The LLM judge sees
  exactly the same text, which keeps the comparison fair.
- Never look up the test outcome or the run's model before you finish labeling. The app hides both.

## Optional: check your own reliability

A week after finishing, label the first 20 items again into a separate file:

```bash
python -m agent_judge label --labels data/labels_pass2.jsonl --n 20
```

Then compare the two passes:

```python
from agent_judge.data import latest_by_id, load_jsonl
from agent_judge.metrics import agreement_table

p1 = latest_by_id(load_jsonl("data/labels.jsonl"))
p2 = latest_by_id(load_jsonl("data/labels_pass2.jsonl"))
print({d: round(v["kappa"], 2) for d, v in agreement_table(p1, p2)["dims"].items()})
```

That is your intra-rater kappa: a realistic ceiling for how well any judge can agree with you.
