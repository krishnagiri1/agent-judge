"""Public Gradio demo: score a coding-agent transcript with the LLM judge.

Deploys as-is to Hugging Face Spaces (see README). Configure the default judge with
the JUDGE_PROVIDER / JUDGE_MODEL environment variables and add the matching API key
as a Space secret.
"""

from __future__ import annotations

import os
from pathlib import Path

from .data import load_trajectories
from .judge import POINTWISE_PROMPTS, SYSTEM_PROMPT, _call_with_parse, build_pointwise_prompt, parse_pointwise
from .llm import make_client
from .render import truncate_middle
from .rubric import DIMENSIONS, rubric_text

ROOT = Path(__file__).resolve().parent.parent


def _load_examples() -> list:
    for p in (ROOT / "data" / "sample.jsonl", ROOT / "examples" / "examples.jsonl"):
        if p.exists():
            return load_trajectories(p)[:50]
    return []


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


def build_app():
    import gradio as gr

    examples = _load_examples()
    by_label = {f"{t.instance_id} ({t.id.split('::')[-1]})": t for t in examples}

    def run(source, example, pasted, provider, model, prompt_version):
        try:
            client = make_client(provider, model)
        except Exception as e:
            return f"⚠️ Could not create the {provider} client: {e}"
        if source == "Example from the dataset":
            if example not in by_label:
                return "Pick an example first."
            t = by_label[example]
            user = build_pointwise_prompt(t, prompt_version)
            truth = t.resolved
        else:
            if not (pasted or "").strip():
                return "Paste a transcript first."
            user = POINTWISE_PROMPTS[prompt_version].format(rubric=rubric_text(),
                                                            transcript=truncate_middle(pasted, 28_000))
            truth = None
        try:
            parsed, _, err = _call_with_parse(client, user, parse_pointwise)
        except Exception as e:
            return f"⚠️ Judge call failed: {type(e).__name__}: {e}"
        return _format(parsed, err, truth)

    def preview(example):
        from .render import render_transcript

        return render_transcript(by_label[example]) if example in by_label else ""

    report = ROOT / "results" / "report.md"
    chart = ROOT / "results" / "kappa_by_dimension.png"

    with gr.Blocks(title="Agent transcript judge") as app:
        gr.Markdown("# LLM-as-Judge for coding-agent transcripts\n"
                    "Scores a SWE-agent run on correctness, tool use, and reasoning, and estimates whether "
                    "its patch passes the hidden tests.")
        with gr.Tab("Judge a transcript"):
            with gr.Row():
                with gr.Column():
                    source = gr.Radio(["Example from the dataset", "Paste my own"],
                                      value="Example from the dataset" if examples else "Paste my own", label="Input")
                    example = gr.Dropdown(list(by_label), label="Example", value=next(iter(by_label), None))
                    pasted = gr.Textbox(label="Transcript text", lines=10)
                    with gr.Row():
                        provider = gr.Dropdown(["openai", "anthropic", "ollama", "mock"],
                                               value=os.environ.get("JUDGE_PROVIDER", "mock"), label="Provider")
                        model = gr.Textbox(value=os.environ.get("JUDGE_MODEL", "heuristic"), label="Model")
                    prompt_version = gr.Radio(list(POINTWISE_PROMPTS), value="v2", label="Prompt")
                    go = gr.Button("Judge", variant="primary")
                with gr.Column():
                    result = gr.Markdown()
                    shown = gr.Textbox(label="Transcript as the judge sees it", lines=20, interactive=False)
            example.change(preview, example, shown)
            app.load(preview, example, shown)
            go.click(run, [source, example, pasted, provider, model, prompt_version], result)
        with gr.Tab("Results"):
            if report.exists():
                gr.Markdown(report.read_text(encoding="utf-8"))
                if chart.exists():
                    gr.Image(str(chart), show_label=False)
            else:
                gr.Markdown("Run `python -m agent_judge report` to generate results.")
        with gr.Tab("Rubric"):
            gr.Markdown(f"```\n{rubric_text()}\n```\n\nSystem prompt:\n\n> {SYSTEM_PROMPT}")
    return app
