"""Command line entry point: python -m agent_judge <command> ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import download_shards, load_trajectories, save_trajectories, sample_trajectories

DEFAULT_SAMPLE = "data/sample.jsonl"
DEFAULT_LABELS = "data/labels.jsonl"


def _add_judge_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", required=True, choices=["openai", "anthropic", "ollama", "mock"])
    p.add_argument("--model", required=True, help="Model name, e.g. an Ollama tag like qwen2.5:14b")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL or Ollama host")
    p.add_argument("--num-ctx", type=int, default=16384, help="Ollama context window")
    p.add_argument("--workers", type=int, default=4, help="Parallel requests (use 1 for a local Ollama model)")
    p.add_argument("--sample", default=DEFAULT_SAMPLE)
    p.add_argument("--limit", type=int, default=None, help="Only use the first N transcripts")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--max-chars", type=int, default=None, help="Transcript render budget (characters)")


def _client(a):
    from .llm import make_client

    kw = {"num_ctx": a.num_ctx} if a.provider == "ollama" else {}
    return make_client(a.provider, a.model, base_url=a.base_url, cache=not a.no_cache, **kw)


def _render_kwargs(a):
    return {"max_chars": a.max_chars} if a.max_chars else None


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "-" for c in s).strip("-")


def cmd_sample(a):
    paths = [Path(p) for p in a.parquet] if a.parquet else download_shards(a.shards, a.cache_dir)
    trajs = sample_trajectories(paths, a.n_instances, seed=a.seed)
    save_trajectories(trajs, a.out)
    n_res = sum(t.resolved for t in trajs)
    print(f"Saved {len(trajs)} transcripts ({n_res} passed tests, {len(trajs) - n_res} failed) "
          f"from {a.n_instances} issues to {a.out}")


def cmd_label(a):
    from .labeling_app import build_app

    build_app(a.sample, a.labels, limit=a.n).launch(server_port=a.port)


def cmd_judge(a):
    from .judge import judge_many

    trajs = load_trajectories(a.sample)[: a.limit]
    client = _client(a)
    out = a.out or f"results/judge_{_slug(client.name)}_{a.prompt}.jsonl"
    res = judge_many(client, trajs, a.prompt, out, workers=a.workers, render_kwargs=_render_kwargs(a))
    errs = sum(1 for r in res if r.get("error"))
    print(f"Judged {len(res)} transcripts -> {out} ({errs} parse/API errors)")


def cmd_position(a):
    from .bias import make_pairs, run_position_bias, summarize_position_bias

    trajs = load_trajectories(a.sample)
    pairs = make_pairs(trajs)[: a.n_pairs]
    client = _client(a)
    tag = f"{_slug(client.name)}_{a.prompt}"
    recs = run_position_bias(client, pairs, a.prompt, f"results/position_{tag}.jsonl", workers=a.workers,
                             render_kwargs=_render_kwargs(a))
    summary = summarize_position_bias(recs)
    Path(f"results/position_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_verbosity(a):
    from .bias import run_verbosity_bias

    trajs = load_trajectories(a.sample)[: a.n]
    client = _client(a)
    summary = run_verbosity_bias(client, trajs, a.prompt, "results", workers=a.workers,
                                 render_kwargs=_render_kwargs(a))
    tag = f"{_slug(client.name)}_{a.prompt}"
    Path(f"results/verbosity_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_report(a):
    from .report import build_report

    print(build_report(a.sample, a.labels, a.results, mlflow=not a.no_mlflow))


def cmd_demo(a):
    from .demo_app import build_app

    build_app().launch(server_port=a.port)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agent_judge", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sample", help="Download shards and sample balanced transcripts")
    p.add_argument("--n-instances", type=int, default=250, help="Issues to sample (2 transcripts each)")
    p.add_argument("--shards", type=int, default=5, help="Parquet shards to download (~90 MB each)")
    p.add_argument("--parquet", nargs="*", help="Use local parquet files instead of downloading")
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=DEFAULT_SAMPLE)
    p.set_defaults(fn=cmd_sample)

    p = sub.add_parser("label", help="Open the hand-labeling app")
    p.add_argument("--sample", default=DEFAULT_SAMPLE)
    p.add_argument("--labels", default=DEFAULT_LABELS)
    p.add_argument("--n", type=int, default=50, help="How many transcripts to label")
    p.add_argument("--port", type=int, default=7860)
    p.set_defaults(fn=cmd_label)

    p = sub.add_parser("judge", help="Score transcripts with an LLM judge")
    _add_judge_args(p)
    p.add_argument("--prompt", default="v2", choices=["v1", "v2"])
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_judge)

    p = sub.add_parser("position-bias", help="Pairwise position-bias experiment")
    _add_judge_args(p)
    p.add_argument("--prompt", default="baseline", choices=["baseline", "debiased"])
    p.add_argument("--n-pairs", type=int, default=100)
    p.set_defaults(fn=cmd_position)

    p = sub.add_parser("verbosity-bias", help="Padding (verbosity) bias experiment")
    _add_judge_args(p)
    p.add_argument("--prompt", default="v2", choices=["v1", "v2"])
    p.add_argument("--n", type=int, default=100)
    p.set_defaults(fn=cmd_verbosity)

    p = sub.add_parser("report", help="Build results/report.md from all runs")
    p.add_argument("--sample", default=DEFAULT_SAMPLE)
    p.add_argument("--labels", default=DEFAULT_LABELS)
    p.add_argument("--results", default="results")
    p.add_argument("--no-mlflow", action="store_true", help="Skip logging runs to MLflow")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("demo", help="Launch the public demo app")
    p.add_argument("--port", type=int, default=7861)
    p.set_defaults(fn=cmd_demo)

    a = ap.parse_args(argv)
    Path("results").mkdir(exist_ok=True)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
