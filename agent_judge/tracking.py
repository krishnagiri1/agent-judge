"""Log experiment results to MLflow.

Each judge run (judge model x prompt) becomes one MLflow run with its parameters and
metrics, and the bias experiments become their own runs. Browse them with `mlflow ui`.
"""

from __future__ import annotations

import math
from pathlib import Path

EXPERIMENT = "agent-judge"


def _num(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _log_metrics(mlflow, metrics: dict) -> None:
    clean = {k: v for k, v in ((k, _num(v)) for k, v in metrics.items()) if v is not None}
    if clean:
        mlflow.log_metrics(clean)


def log_to_mlflow(metrics: dict, results_dir: str | Path = "results", tracking_uri: str | None = None) -> int:
    """Log the metrics dict produced by `build_report`. Returns the number of runs logged."""
    import mlflow

    results_dir = Path(results_dir)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    n = 0

    for stem, r in (metrics.get("runs") or {}).items():
        with mlflow.start_run(run_name=f"judge:{r['judge']}:{r['prompt']}"):
            mlflow.set_tag("experiment_type", "judge")
            mlflow.log_params({"judge": r["judge"], "prompt": r["prompt"],
                               "n_transcripts": metrics.get("n_transcripts"),
                               "n_human_labels": metrics.get("n_human_labels")})
            gt = r.get("ground_truth_all") or {}
            m = {
                "auroc_resolved_probability": gt.get("auroc_resolved_probability"),
                "auroc_correctness": gt.get("auroc_correctness"),
                "auroc_overall": gt.get("auroc_overall"),
                "parse_error_rate": r.get("parse_error_rate"),
            }
            for dim, d in ((r.get("agreement") or {}).get("dims") or {}).items():
                m[f"kappa_{dim}"] = d.get("kappa")
                m[f"spearman_{dim}"] = d.get("spearman")
            _log_metrics(mlflow, m)
            n += 1

    for s in metrics.get("position_bias") or []:
        with mlflow.start_run(run_name=f"position_bias:{s['judge']}:{s['prompt']}"):
            mlflow.set_tag("experiment_type", "position_bias")
            mlflow.log_params({"judge": s["judge"], "prompt": s["prompt"], "n_pairs": s["n_pairs"]})
            _log_metrics(mlflow, {k: s.get(k) for k in ("inconsistency_rate", "first_position_win_rate", "tie_rate",
                                                        "accuracy_single_order", "accuracy_swap_aggregated")})
            n += 1

    for s in metrics.get("verbosity_bias") or []:
        with mlflow.start_run(run_name=f"verbosity_bias:{s['judge']}:{s['prompt']}"):
            mlflow.set_tag("experiment_type", "verbosity_bias")
            mlflow.log_params({"judge": s["judge"], "prompt": s["prompt"], "n": s["n"]})
            _log_metrics(mlflow, {f"delta_{d}": v.get("mean_delta") for d, v in s["dims"].items()})
            n += 1

    report = results_dir / "report.md"
    if report.exists() and n:
        with mlflow.start_run(run_name="report"):
            mlflow.set_tag("experiment_type", "report")
            mlflow.log_artifact(str(report))
            chart = results_dir / "kappa_by_dimension.png"
            if chart.exists():
                mlflow.log_artifact(str(chart))
    return n
