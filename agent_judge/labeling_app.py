"""Gradio app for hand-labeling transcripts with the judge's rubric.

Blind by design: you never see the test outcome or which model produced the run, and
you see exactly the rendered text the LLM judge sees. Labels append to a JSONL file;
re-labeling an item just adds a newer record, which wins.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .data import append_jsonl, latest_by_id, load_jsonl, load_trajectories
from .render import render_transcript
from .rubric import DIMENSIONS, SCORE_DIMENSIONS, rubric_text


def build_app(sample_path: str | Path, labels_path: str | Path, limit: int = 50):
    import gradio as gr

    trajs = load_trajectories(sample_path)[:limit]
    by_id = {t.id: t for t in trajs}
    order = [t.id for t in trajs]

    def labeled() -> dict:
        return {i: r for i, r in latest_by_id(load_jsonl(labels_path)).items() if r.get("scores")}

    def next_id(after: str | None = None) -> str | None:
        done = labeled()
        start = order.index(after) + 1 if after in order else 0
        for i in order[start:] + order[:start]:
            if i not in done:
                return i
        return None

    def view(item_id: str | None):
        n_done = len([i for i in labeled() if i in by_id])
        progress = f"**{n_done} / {len(order)} labeled**"
        if item_id is None:
            return item_id, progress + " · all done 🎉", "", *[None] * len(DIMENSIONS), 50, ""
        t = by_id[item_id]
        prev = labeled().get(item_id, {})
        ps = prev.get("scores", {})
        return (item_id, f"{progress} · item `{order.index(item_id) + 1}`", render_transcript(t),
                *[ps.get(d) for d in DIMENSIONS], prev.get("resolved_probability", 50), prev.get("notes", ""))

    def save(item_id, *vals):
        *scores, prob, notes = vals
        if item_id is None:
            return view(None)
        if any(s is None for s in scores):
            gr.Warning("Score every dimension before saving.")
            return view(item_id)
        append_jsonl({
            "id": item_id,
            "scores": {d: int(s) for d, s in zip(DIMENSIONS, scores)},
            "resolved_probability": float(prob),
            "notes": notes or "",
            "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, labels_path)
        return view(next_id(item_id))

    with gr.Blocks(title="Agent transcript labeling") as app:
        current = gr.State(next_id())
        gr.Markdown("# Agent transcript labeling\nScore each run against the rubric. "
                    "You are blind to the test outcome and the model that produced it.")
        with gr.Accordion("Rubric", open=False):
            gr.Markdown(f"```\n{rubric_text()}\n```")
        progress = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=3):
                transcript = gr.Textbox(label="Transcript (identical to what the judge sees)", lines=32,
                                        max_lines=60, interactive=False)
            with gr.Column(scale=1, min_width=280):
                radios = [gr.Radio([1, 2, 3, 4, 5], label=f"{d} – {SCORE_DIMENSIONS[d]['question']}")
                          for d in DIMENSIONS]
                prob = gr.Slider(0, 100, value=50, step=5, label="Probability the patch passes the tests (%)")
                notes = gr.Textbox(label="Notes (optional)", lines=3)
                save_btn = gr.Button("Save & next", variant="primary")
                skip_btn = gr.Button("Skip")

        outputs = [current, progress, transcript, *radios, prob, notes]
        app.load(view, inputs=current, outputs=outputs)
        save_btn.click(save, inputs=[current, *radios, prob, notes], outputs=outputs)
        skip_btn.click(lambda i: view(next_id(i)), inputs=current, outputs=outputs)
    return app
