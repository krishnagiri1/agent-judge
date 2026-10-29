"""End-to-end pipeline with the mock judge: no network, no API key."""

import json

from agent_judge.bias import make_pairs, run_position_bias, run_verbosity_bias, summarize_position_bias
from agent_judge.data import append_jsonl, load_jsonl, save_trajectories
from agent_judge.judge import judge_many
from agent_judge.llm import make_client
from agent_judge.report import build_report


def test_judge_many_writes_and_resumes(tmp_path, trajs):
    client = make_client("mock", "h", cache_path=tmp_path / "c.sqlite")
    out = tmp_path / "judge.jsonl"
    res = judge_many(client, trajs, "v2", out, workers=4, progress=False)
    assert len(res) == len(trajs) and all(r.get("scores") for r in res)
    judge_many(client, trajs, "v2", out, workers=4, progress=False)  # resume: nothing new
    assert len(load_jsonl(out)) == len(trajs)


def test_mock_judge_separates_resolved(tmp_path, trajs):
    client = make_client("mock", "h", cache=False)
    res = judge_many(client, trajs, "v1", tmp_path / "j.jsonl", progress=False)
    truth = {t.id: t.resolved for t in trajs}
    pos = [r["resolved_probability"] for r in res if truth[r["id"]]]
    neg = [r["resolved_probability"] for r in res if not truth[r["id"]]]
    assert min(pos) > max(neg)


def test_position_bias_detected(tmp_path, trajs):
    pairs = make_pairs(trajs)
    assert len(pairs) == 10
    fair = make_client("mock", "fair", cache=False)
    biased = make_client("mock", "biased", cache=False, position_bias=0.6)
    s_fair = summarize_position_bias(run_position_bias(fair, pairs, "baseline", tmp_path / "f.jsonl"))
    s_bias = summarize_position_bias(run_position_bias(biased, pairs, "baseline", tmp_path / "b.jsonl"))
    assert s_fair["inconsistency_rate"] == 0 and s_fair["accuracy_swap_aggregated"] == 1.0
    assert s_bias["first_position_win_rate"] > 0.6
    assert s_bias["inconsistency_rate"] > 0


def test_verbosity_bias_detected(tmp_path, trajs):
    biased = make_client("mock", "wordy", cache=False, verbosity_bias=1)
    s = run_verbosity_bias(biased, trajs, "v2", tmp_path)
    assert s["n"] == len(trajs)
    assert s["dims"]["overall"]["mean_delta"] > 0
    assert s["dims"]["correctness"]["mean_delta"] == 0


def test_report_end_to_end(tmp_path, trajs, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_trajectories(trajs, "data/sample.jsonl")
    client = make_client("mock", "h", cache=False)
    judge_many(client, trajs, "v2", "results/judge_mock_v2.jsonl", progress=False)
    for r in load_jsonl("results/judge_mock_v2.jsonl")[:12]:  # pretend a human agreed, with some noise
        s = dict(r["scores"])
        s["overall"] = max(1, min(5, s["overall"] + (1 if hash(r["id"]) % 3 == 0 else 0)))
        append_jsonl({"id": r["id"], "scores": s, "resolved_probability": 50}, "data/labels.jsonl")
    pairs = make_pairs(trajs)
    summ = summarize_position_bias(run_position_bias(client, pairs, "baseline", "results/position_x.jsonl"))
    (tmp_path / "results/position_x.json").write_text(json.dumps(summ))
    report = build_report("data/sample.jsonl", "data/labels.jsonl", "results")
    assert "Judge vs. human agreement" in report and "Position bias" in report
    assert (tmp_path / "results/metrics.json").exists()
    assert (tmp_path / "results/kappa_by_dimension.png").exists()


def test_report_without_human_labels(tmp_path, trajs, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_trajectories(trajs, "data/sample.jsonl")
    judge_many(make_client("mock", "h", cache=False), trajs, "v2", "results/judge_mock_v2.jsonl", progress=False)
    report = build_report("data/sample.jsonl", "data/labels.jsonl", "results")
    assert report.index("Predicting the hidden test outcome") < report.index("Judge vs. human agreement")
    assert "No human labels yet" in report


def test_mlflow_logging(tmp_path, trajs, monkeypatch):
    import pytest

    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    save_trajectories(trajs, "data/sample.jsonl")
    judge_many(make_client("mock", "h", cache=False), trajs, "v2", "results/judge_mock_v2.jsonl", progress=False)
    report = build_report("data/sample.jsonl", "data/labels.jsonl", "results", mlflow=True)
    assert "Logged" in report
    runs = mlflow.search_runs(experiment_names=["agent-judge"])
    assert "metrics.auroc_resolved_probability" in runs.columns


def test_export_site(tmp_path, trajs, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from agent_judge.site import export_site

    save_trajectories(trajs, "data/sample.jsonl")
    judge_many(make_client("mock", "h", cache=False), trajs, "v2", "results/judge_mock_v2.jsonl", progress=False)
    page = export_site("data/sample.jsonl", "data/labels.jsonl", "results", "docs")
    js = (tmp_path / "docs" / "data.js").read_text()
    assert page.exists() and js.startswith("window.AGENT_JUDGE_DATA = ")
    data = json.loads(js[len("window.AGENT_JUDGE_DATA = "):].rstrip().rstrip(";"))
    assert len(data["items"]) == len(trajs) and data["items"][0]["verdicts"]
