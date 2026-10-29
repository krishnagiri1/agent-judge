"""Build results/report.md and results/metrics.json from all saved runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .data import latest_by_id, load_jsonl, load_trajectories
from .metrics import agreement_table, fmt, ground_truth_table, pct
from .rubric import DIMENSIONS


def display_name(judge: str) -> str:
    """Readable rater names for the report (the raw name stays in metrics.json and MLflow)."""
    if judge.startswith("mock:heuristic"):
        return "Heuristic baseline (no LLM)"
    return judge


def _runs(results_dir: Path) -> list[Path]:
    return sorted(p for p in results_dir.glob("judge_*.jsonl"))


def _jsonable(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def build_report(sample_path: str | Path, labels_path: str | Path, results_dir: str | Path = "results",
                 mlflow: bool = False) -> str:
    results_dir = Path(results_dir)
    trajs = load_trajectories(sample_path)
    truth = {t.id: t.resolved for t in trajs}
    human = {i: r for i, r in latest_by_id(load_jsonl(labels_path)).items() if r.get("scores")}
    metrics: dict = {"n_transcripts": len(trajs), "n_issues": len({t.instance_id for t in trajs}),
                     "n_human_labels": len(human), "runs": {}, "position_bias": [], "verbosity_bias": []}

    L = ["# Results", ""]
    L.append(f"- Transcripts judged: **{len(trajs)}** public SWE-agent runs on **{metrics['n_issues']}** real GitHub "
             f"issues (`nebius/SWE-agent-trajectories`, CC-BY-4.0), balanced 50/50 between runs that passed and "
             f"failed the repository's tests.")
    L.append(f"- Human labels: **{len(human)}** transcripts hand-scored with the same rubric and the same rendered "
             f"text the judge sees (blind to the test outcome and the agent model).")
    L.append("")

    # ---- per-run metrics ---------------------------------------------------
    run_paths = _runs(results_dir)
    if run_paths:
        A = ["## Judge vs. human agreement", "",
             "Quadratic-weighted Cohen's kappa on the 1-5 scale (95% bootstrap CI for overall).", "",
             "| Judge | Prompt | n | " + " | ".join(f"κ {d}" for d in DIMENSIONS) +
             " | Spearman (overall) | Within ±1 (overall) | Parse errors |",
             "|---|---|---|" + "---|" * len(DIMENSIONS) + "---|---|---|"]
        G = ["## Predicting the hidden test outcome (primary result)", "",
             "AUROC of each rater's `resolved_probability` and `correctness` against whether the patch "
             "actually passed the repository's tests (0.5 = chance, 1.0 = perfect), with a 95% bootstrap CI.", "",
             "| Rater | Prompt | n (all) | AUROC P(resolved) | AUROC correctness | Parse errors | n (labeled) | AUROC P(resolved), labeled subset |",
             "|---|---|---|---|---|---|---|---|"]
        for p in run_paths:
            latest = latest_by_id(load_jsonl(p))
            if not latest:
                continue
            ok = {i: r for i, r in latest.items() if r.get("scores")}
            judge = next(iter(latest.values())).get("judge", p.stem)
            prompt = next(iter(latest.values())).get("prompt", "?")
            err_rate = 1 - len(ok) / len(latest)
            ag = agreement_table(human, ok)
            gt = ground_truth_table(ok, truth)
            gt_h = ground_truth_table({i: ok[i] for i in ok if i in human}, truth)
            metrics["runs"][p.stem] = {"judge": judge, "prompt": prompt, "parse_error_rate": err_rate,
                                       "agreement": ag, "ground_truth_all": gt, "ground_truth_labeled": gt_h}
            gci = gt.get("auroc_resolved_probability_ci95") or (float("nan"), float("nan"))
            G.append(f"| {display_name(judge)} | {prompt} | {gt['n']} | {fmt(gt['auroc_resolved_probability'])} "
                     f"[{fmt(gci[0])}, {fmt(gci[1])}] | "
                     f"{fmt(gt['auroc_correctness'])} | {pct(err_rate)} | {gt_h['n']} | "
                     f"{fmt(gt_h['auroc_resolved_probability'])} |")
            o = ag["dims"]["overall"]
            ci = o["kappa_ci95"]
            A.append(f"| {display_name(judge)} | {prompt} | {ag['n']} | " +
                     " | ".join(fmt(ag["dims"][d]["kappa"]) for d in DIMENSIONS[:-1]) +
                     f" | {fmt(o['kappa'])} [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(o['spearman'])} | "
                     f"{pct(o['within_1'])} | {pct(err_rate)} |")
        if human:
            hg = ground_truth_table(human, truth)
            metrics["human_ground_truth"] = hg
            G.append(f"| **Human (you)** | – | – | – | – | – | {hg['n']} | {fmt(hg['auroc_resolved_probability'])} |")
        L += G + [""]
        if human:
            L += A + [""]
        else:
            L += ["## Judge vs. human agreement", "", "_No human labels yet: run `python -m agent_judge label`._", ""]

    # ---- position bias ---------------------------------------------------
    pos = []
    for p in sorted(results_dir.glob("position_*.json")):
        pos.append(json.loads(p.read_text()))
    if pos:
        metrics["position_bias"] = pos
        L += ["## Position bias (pairwise, both orders)", "",
              "Each pair is one passing and one failing run on the same issue, judged twice with the order swapped.", "",
              "| Judge | Prompt | Pairs | Verdict flips with order | 'A' win rate | Accuracy (single order) | Accuracy (swap-aggregated) |",
              "|---|---|---|---|---|---|---|"]
        for s in pos:
            L.append(f"| {display_name(s['judge'])} | {s['prompt']} | {s['n_pairs']} | {pct(s['inconsistency_rate'])} | "
                     f"{pct(s['first_position_win_rate'])} | {pct(s['accuracy_single_order'])} | {pct(s['accuracy_swap_aggregated'])} |")
        by_judge: dict[str, dict] = {}
        for s in pos:
            by_judge.setdefault(s["judge"], {})[s["prompt"]] = s
        for judge, d in by_judge.items():
            if "baseline" in d and "debiased" in d and d["baseline"]["inconsistency_rate"]:
                b, a = d["baseline"]["inconsistency_rate"], d["debiased"]["inconsistency_rate"]
                L.append("")
                change = "reduction" if a < b else "increase"
                L.append(f"**{display_name(judge)}:** the debiased prompt changed the order-flip rate from {pct(b)} "
                         f"to {pct(a)} ({pct(abs(b - a) / b)} relative {change}).")
        L.append("")

    # ---- verbosity bias --------------------------------------------------
    verb = [json.loads(p.read_text()) for p in sorted(results_dir.glob("verbosity_*.json"))]
    if verb:
        metrics["verbosity_bias"] = verb
        L += ["## Verbosity bias (content-free narration added to every agent turn)", "",
              "| Judge | Prompt | n | Δ overall | Δ reasoning | Δ correctness | Δ P(resolved) | Share scored higher (overall) | Wilcoxon p (overall) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for s in verb:
            d = s["dims"]
            L.append(f"| {display_name(s['judge'])} | {s['prompt']} | {s['n']} | {fmt(d['overall']['mean_delta'], 2)} | "
                     f"{fmt(d['reasoning']['mean_delta'], 2)} | {fmt(d['correctness']['mean_delta'], 2)} | "
                     f"{fmt(d['resolved_probability']['mean_delta'], 1)} | {pct(d['overall']['share_up'])} | "
                     f"{fmt(d['overall']['wilcoxon_p'], 3)} |")
        L.append("")

    report = "\n".join(L)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "report.md").write_text(report, encoding="utf-8")
    (results_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
    _plot(metrics, results_dir)
    if mlflow:
        from .tracking import log_to_mlflow

        n = log_to_mlflow(_jsonable(metrics), results_dir)
        report += f"\n\n_Logged {n} runs to MLflow (experiment `agent-judge`); view with `mlflow ui`._"
    return report


def _plot(metrics: dict, results_dir: Path) -> None:
    """Bar chart of judge-human kappa by dimension for each run (skipped if matplotlib is missing)."""
    runs = metrics.get("runs") or {}
    if not runs or not metrics.get("n_human_labels"):
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(len(runs), 1)
    for k, (name, r) in enumerate(runs.items()):
        vals = [r["agreement"]["dims"][d]["kappa"] for d in DIMENSIONS]
        vals = [0 if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in vals]
        xs = [i + k * width for i in range(len(DIMENSIONS))]
        ax.bar(xs, vals, width, label=f"{r['judge']} · {r['prompt']}")
    ax.set_xticks([i + width * (len(runs) - 1) / 2 for i in range(len(DIMENSIONS))])
    ax.set_xticklabels(DIMENSIONS)
    ax.set_ylabel("Weighted Cohen's κ vs. human")
    ax.set_ylim(0, 1)
    ax.axhline(0.6, color="grey", lw=0.8, ls="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(results_dir / "kappa_by_dimension.png", dpi=150)
    plt.close(fig)
