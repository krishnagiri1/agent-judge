"""Publish the demo to a Hugging Face Space in one command.

    python -c "from huggingface_hub import login; login()"     # once: paste a write token
    python scripts/deploy_space.py

It creates (or updates) https://huggingface.co/spaces/<your-username>/agent-judge with only
what the demo needs: the code, the sampled transcripts, your labels, and the results. The
Space gets its own README header and a minimal requirements file.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPACE_REQUIREMENTS = "openai>=1.40\nanthropic>=0.34\n"

SPACE_README = """---
title: Agent Judge
emoji: ⚖️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
license: mit
short_description: How well can an LLM grade an AI coding agent?
---

# LLM-as-Judge for coding-agent transcripts

Browse 500 real SWE-agent runs with the verdicts of a local qwen2.5:14b judge, a no-LLM
heuristic baseline, human labels, and the actual test outcome.

Code, method, and full results: {github}
"""


def stage(dst: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "agent_judge", dst / "agent_judge", ignore=ignore)
    shutil.copy2(ROOT / "app.py", dst / "app.py")
    for rel in ("data/sample.jsonl", "data/labels.jsonl"):
        if (ROOT / rel).exists():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dst / rel)
    shutil.copytree(ROOT / "examples", dst / "examples")
    results = dst / "results"
    results.mkdir()
    for p in (ROOT / "results").iterdir():
        if p.name.startswith("judge_") or p.suffix in (".md", ".png", ".json"):
            shutil.copy2(p, results / p.name)


def main() -> None:
    import gradio
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", default=None, help="username/space-name (default: <you>/agent-judge)")
    ap.add_argument("--github", default="https://github.com/krishnagiri1/agent-judge")
    a = ap.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = a.space or f"{user}/agent-judge"
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp)
        stage(dst)
        (dst / "requirements.txt").write_text(SPACE_REQUIREMENTS)
        (dst / "README.md").write_text(SPACE_README.format(gradio_version=gradio.__version__, github=a.github),
                                       encoding="utf-8")
        api.upload_folder(folder_path=str(dst), repo_id=repo_id, repo_type="space",
                          commit_message="Deploy demo with saved verdicts")
    print(f"Done. Your Space is building at https://huggingface.co/spaces/{repo_id}")
    print("The first build takes a few minutes; refresh the page until the app appears.")


if __name__ == "__main__":
    main()
