"""Public Gradio demo.

Tab 1 shows the saved verdicts from the experiment runs (no API key or GPU needed), next to
the actual test outcome and, where available, the human label. Tab 2 judges a transcript live
with any configured provider. Configure the live judge with JUDGE_PROVIDER / JUDGE_MODEL and
the matching API key (as Space secrets on Hugging Face).
"""

from __future__ import annotations

import os
from pathlib import Path

from .data import latest_by_id, load_jsonl, load_trajectories
from .judge import POINTWISE_PROMPTS, SYSTEM_PROMPT, _call_with_parse, build_pointwise_prompt, parse_pointwise
from .llm import make_client
from .render import render_transcript, truncate_middle
from .rubric import DIMENSIONS, rubric_text

ROOT = Path(__file__).resolve().parent.parent

FILTERS = ["All transcripts", "Hand-labeled (50)", "Judge was wrong", "Judge was right"]


def _rater_name(judge: str, prompt: str) -> str:
    if judge.startswith("mock:heuristic"):
        return "Heuristic baseline (no LLM)"
    return f"{judge.split(':', 1)[-1]} · prompt {prompt}"


def _load_trajectories() -> list:
    for p in (ROOT / "data" / "sample.jsonl", ROOT / "examples" / "examples.jsonl"):
        if p.exists():
            return load_trajectories(p)
    return []


def _load_saved_runs() -> list[tuple[str, dict]]:
    """[(rater name, {transcript id: record})] from results/judge_*.jsonl; LLM runs first."""
    runs = []
    for p in sorted((ROOT / "results").glob("judge_*.jsonl")):
        recs = {i: r for i, r in latest_by_id(load_jsonl(p)).items() if r.get("scores")}
        if recs:
            first = next(iter(recs.values()))
            runs.append((_rater_name(first.get("judge", p.stem), first.get("prompt", "?")), recs))
    runs.sort(key=lambda x: x[0].startswith("Heuristic"))
    return runs


def _format(parsed: dict | None, err: str | None, truth: bool | None) -> str:
    if parsed is None:
        return f"⚠️ The judge's reply could not be parsed: {err}"
    rows = "\n".join(f"| {d} | {parsed['scores'][d]} / 5 |" for d in DIMENSIONS)
    out = (f"| Dimension | Score |\n|---|---|\n{rows}\n| P(patch passes tests) | "
           f"{parsed['resolved_probability']:.0f}% |\n\n**Rationale:** {parsed['rationale']}")
    if parsed.get("analysis"):
        out += f"\n\n<details><summary>Judge's analysis</summary>\n\n{parsed['analysis']}\n\n</details>"
    if truth is not None:
        out += f"\n\n**Actual test outcome:** {'✅ passed' if truth else '❌ failed'}"
    return out


def _verdict_table(tid: str, truth: bool, runs: list, human: dict) -> str:
    raters = [(name, recs[tid]) for name, recs in runs if tid in recs]
    if tid in human:
        raters.append(("**Human label**", human[tid]))
    if not raters:
        return "_No saved verdicts for this transcript._"
    head = "| Rater | " + " | ".join(DIMENSIONS) + " | P(pass) | Right about the outcome? |"
    sep = "|---|" + "---|" * (len(DIMENSIONS) + 2)
    lines = [head, sep]
    for name, r in raters:
        p = r.get("resolved_probability")
        right = "–" if p is None or p == 50 else ("✅" if (p > 50) == truth else "❌")
        prob = "–" if p is None else f"{p:.0f}%"
        scores = " | ".join(str(r["scores"].get(d, "–")) for d in DIMENSIONS)
        lines.append(f"| {name} | {scores} | {prob} | {right} |")
    out = f"**Actual test outcome: {'✅ the patch passed' if truth else '❌ the patch failed'}**\n\n" + "\n".join(lines)
    rationale = [(n, r.get("rationale")) for n, r in raters if r.get("rationale") and not n.startswith("Heuristic")]
    for n, text in rationale[:2]:
        out += f"\n\n**{n} rationale:** {text}"
    return out


def build_app():
    import gradio as gr

    trajs = _load_trajectories()
    runs = _load_saved_runs()
    human = {i: r for i, r in latest_by_id(load_jsonl(ROOT / "data" / "labels.jsonl")).items() if r.get("scores")}
    best = runs[0][1] if runs else {}

    def label(i, t):
        return f"#{i + 1:03d} · {t.instance_id}"

    by_label = {label(i, t): t for i, t in enumerate(trajs)}

    def choices(flt: str) -> list[str]:
        out = []
        for lab, t in by_label.items():
            p = best.get(t.id, {}).get("resolved_probability")
            wrong = p is not None and p != 50 and (p > 50) != t.resolved
            if (flt == "Hand-labeled (50)" and t.id not in human) or \
               (flt == "Judge was wrong" and not wrong) or \
               (flt == "Judge was right" and (p is None or p == 50 or wrong)):
                continue
            out.append(lab)
        return out

    def show_saved(lab):
        if lab not in by_label:
            return "", ""
        t = by_label[lab]
        return _verdict_table(t.id, t.resolved, runs, human), render_transcript(t)

    def refilter(flt):
        c = choices(flt)
        first = c[0] if c else None
        return (gr.update(choices=c, value=first), *show_saved(first))

    def run_live(source, lab, pasted, provider, model, prompt_version):
        try:
            client = make_client(provider, model)
        except Exception as e:
            return f"⚠️ Could not create the {provider} client: {e}"
        if source == "Example from the dataset":
            if lab not in by_label:
                return "Pick an example first."
            t = by_label[lab]
            user, truth = build_pointwise_prompt(t, prompt_version), t.resolved
        else:
            if not (pasted or "").strip():
                return "Paste a transcript first."
            user = POINTWISE_PROMPTS[prompt_version].format(rubric=rubric_text(), transcript=truncate_middle(pasted, 28_000))
            truth = None
        try:
            parsed, _, err = _call_with_parse(client, user, parse_pointwise)
        except Exception as e:
            return f"⚠️ Judge call failed: {type(e).__name__}: {e}"
        return _format(parsed, err, truth)

    report = ROOT / "results" / "report.md"
    first_choices = choices(FILTERS[0])
    first = first_choices[0] if first_choices else None

    with gr.Blocks(title="Agent transcript judge") as app:
        gr.Markdown("# LLM-as-Judge for coding-agent transcripts\n"
                    "How well can an LLM grade an AI coding agent? Each transcript is a real SWE-agent attempt at a "
                    "GitHub issue, with the hidden truth of whether its patch passed the tests. "
                    "[Code and write-up on GitHub](https://github.com/krishnagiri1/agent-judge)")
        with gr.Tab("Saved verdicts"):
            gr.Markdown(f"Verdicts recorded during the experiments ({len(trajs)} transcripts). "
                        "Compare each judge with the actual outcome and the human label.")
            with gr.Row():
                flt = gr.Radio(FILTERS, value=FILTERS[0], label="Show")
                pick = gr.Dropdown(first_choices, value=first, label="Transcript")
            verdicts = gr.Markdown()
            transcript = gr.Textbox(label="Transcript (exactly what the judge saw)", lines=24, interactive=False)
            flt.change(refilter, flt, [pick, verdicts, transcript])
            pick.change(show_saved, pick, [verdicts, transcript])
            app.load(show_saved, pick, [verdicts, transcript])
        with gr.Tab("Judge live"):
            gr.Markdown("Run the judge now. Without an API key configured, only the heuristic baseline (`mock`) works.")
            with gr.Row():
                with gr.Column():
                    source = gr.Radio(["Example from the dataset", "Paste my own"], value="Example from the dataset",
                                      label="Input")
                    lab_live = gr.Dropdown(list(by_label), value=first, label="Example")
                    pasted = gr.Textbox(label="Transcript text", lines=8)
                    with gr.Row():
                        provider = gr.Dropdown(["openai", "anthropic", "ollama", "mock"],
                                               value=os.environ.get("JUDGE_PROVIDER", "mock"), label="Provider")
                        model = gr.Textbox(value=os.environ.get("JUDGE_MODEL", "heuristic"), label="Model")
                    prompt_version = gr.Radio(list(POINTWISE_PROMPTS), value="v1", label="Prompt")
                    go = gr.Button("Judge", variant="primary")
                with gr.Column():
                    result = gr.Markdown()
            go.click(run_live, [source, lab_live, pasted, provider, model, prompt_version], result)
        with gr.Tab("Results"):
            gr.Markdown(report.read_text(encoding="utf-8") if report.exists()
                        else "Run `python -m agent_judge report` to generate results.")
        with gr.Tab("Rubric"):
            gr.Markdown(f"```\n{rubric_text()}\n```\n\nSystem prompt:\n\n> {SYSTEM_PROMPT}")
    return app
