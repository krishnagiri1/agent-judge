"""Bias experiments for LLM judges.

Position bias
    Give the judge two runs on the same issue (one passed the tests, one failed) and ask
    which is better, then ask again with the order swapped. An unbiased judge picks the
    same transcript both times. We report how often the verdict flips with order
    (inconsistency), how often "A" wins, and accuracy against the test outcome, with and
    without swap-aggregation (only trust verdicts that survive the swap).

Verbosity bias
    Judge each transcript twice: as-is, and with content-free narration prepended to every
    agent turn (no actions, outputs, or patch changes). Any systematic score increase on
    the padded version is verbosity bias.
"""

from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .data import Trajectory, append_jsonl, load_jsonl
from .judge import judge_many, judge_pair
from .llm import LLMClient
from .rubric import DIMENSIONS

# ---------------------------------------------------------------------------
# Position bias
# ---------------------------------------------------------------------------


def make_pairs(trajs: list[Trajectory]) -> list[tuple[Trajectory, Trajectory]]:
    """(resolved, unresolved) pairs that share an issue."""
    by_inst: dict[str, dict[bool, Trajectory]] = {}
    for t in trajs:
        by_inst.setdefault(t.instance_id, {}).setdefault(t.resolved, t)
    return [(d[True], d[False]) for d in by_inst.values() if True in d and False in d]


def _chosen_id(rec: dict) -> Optional[str]:
    w = rec.get("winner")
    if w == "A":
        return rec["first"]
    if w == "B":
        return rec["second"]
    return None  # tie or error


def run_position_bias(
    client: LLMClient,
    pairs: list[tuple[Trajectory, Trajectory]],
    prompt: str,
    out_path: str | Path,
    workers: int = 4,
    render_kwargs: Optional[dict] = None,
) -> list[dict]:
    """Judge each pair in both orders; resumable via `out_path`."""
    existing = {(r["first"], r["second"]): r for r in load_jsonl(out_path) if not r.get("error")}
    jobs = []
    for good, bad in pairs:
        for a, b in ((good, bad), (bad, good)):
            if (a.id, b.id) not in existing:
                jobs.append((good, a, b))
    lock = threading.Lock()

    def work(job):
        good, a, b = job
        try:
            rec = judge_pair(client, a, b, prompt, render_kwargs)
        except Exception as e:
            rec = {"first": a.id, "second": b.id, "error": f"{type(e).__name__}: {e}"}
        rec.update({"instance_id": good.instance_id, "resolved_id": good.id, "judge": client.name, "prompt": prompt})
        return rec

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for f in as_completed([pool.submit(work, j) for j in jobs]):
            rec = f.result()
            with lock:
                append_jsonl(rec, out_path)
                if not rec.get("error"):
                    existing[(rec["first"], rec["second"])] = rec
    return list(existing.values())


def summarize_position_bias(records: list[dict]) -> dict:
    by_inst: dict[str, list[dict]] = {}
    for r in records:
        if not r.get("error"):
            by_inst.setdefault(r["instance_id"], []).append(r)
    decisions = [r for rs in by_inst.values() for r in rs]
    ab = [r for r in decisions if r.get("winner") in ("A", "B")]

    def single_score(r):
        c = _chosen_id(r)
        return 0.5 if c is None else float(c == r["resolved_id"])

    complete = [rs for rs in by_inst.values() if len(rs) == 2]
    inconsistent = 0
    agg = []
    for rs in complete:
        c1, c2 = _chosen_id(rs[0]), _chosen_id(rs[1])
        if c1 != c2:
            inconsistent += 1
            agg.append(0.5)
        elif c1 is None:
            agg.append(0.5)
        else:
            agg.append(float(c1 == rs[0]["resolved_id"]))

    n = len(complete)
    return {
        "judge": records[0].get("judge") if records else None,
        "prompt": records[0].get("prompt") if records else None,
        "n_pairs": n,
        "n_decisions": len(decisions),
        "inconsistency_rate": inconsistent / n if n else float("nan"),
        "first_position_win_rate": sum(r["winner"] == "A" for r in ab) / len(ab) if ab else float("nan"),
        "tie_rate": sum(r.get("winner") == "tie" for r in decisions) / len(decisions) if decisions else float("nan"),
        "accuracy_single_order": sum(map(single_score, decisions)) / len(decisions) if decisions else float("nan"),
        "accuracy_swap_aggregated": sum(agg) / len(agg) if agg else float("nan"),
    }


# ---------------------------------------------------------------------------
# Verbosity bias
# ---------------------------------------------------------------------------

FILLER_SENTENCES = [
    "Let me think about this carefully and make sure I am being thorough and methodical here.",
    "It is important to approach this step with rigor so that the final solution is robust and high quality.",
    "I want to be confident that my understanding is complete before proceeding further.",
    "Taking a careful, systematic approach here will help ensure the change is correct and well-reasoned.",
    "I will keep the overall goal in mind and double-check my reasoning as I go.",
    "This is a key step, so I am going to proceed deliberately and pay close attention to detail.",
    "Being precise at this point matters, because small mistakes can have larger consequences later.",
    "I am confident this methodical approach is the best way to make steady, reliable progress.",
]


def make_filler(seed: int = 0, sentences_per_turn: int = 2):
    """Content-free narration: no actions, facts, or claims about the code."""

    def filler(step: int) -> str:
        rng = random.Random(seed * 100_003 + step)
        return " ".join(rng.sample(FILLER_SENTENCES, sentences_per_turn))

    return filler


def run_verbosity_bias(
    client: LLMClient,
    trajs: list[Trajectory],
    prompt_version: str,
    out_dir: str | Path,
    workers: int = 4,
    render_kwargs: Optional[dict] = None,
    seed: int = 0,
) -> dict:
    out_dir = Path(out_dir)
    tag = f"{_slug(client.name)}_{prompt_version}"
    orig = judge_many(client, trajs, prompt_version, out_dir / f"verbosity_{tag}_original.jsonl",
                      workers, render_kwargs, progress=True)
    padded = judge_many(client, trajs, prompt_version, out_dir / f"verbosity_{tag}_padded.jsonl",
                        workers, render_kwargs, filler=make_filler(seed), progress=True)
    return summarize_verbosity_bias(orig, padded, client.name, prompt_version)


def summarize_verbosity_bias(orig: list[dict], padded: list[dict], judge: str = "", prompt: str = "") -> dict:
    from scipy.stats import wilcoxon

    o = {r["id"]: r for r in orig if r.get("scores")}
    p = {r["id"]: r for r in padded if r.get("scores")}
    ids = [i for i in o if i in p]
    out: dict = {"judge": judge, "prompt": prompt, "n": len(ids), "dims": {}}
    fields = DIMENSIONS + ["resolved_probability"]
    for d in fields:
        get = (lambda r: r["resolved_probability"]) if d == "resolved_probability" else (lambda r, d=d: r["scores"][d])
        deltas = [get(p[i]) - get(o[i]) for i in ids]
        nz = [x for x in deltas if x != 0]
        try:
            pval = float(wilcoxon(deltas).pvalue) if len(nz) >= 5 else float("nan")
        except ValueError:
            pval = float("nan")
        out["dims"][d] = {
            "mean_delta": sum(deltas) / len(deltas) if deltas else float("nan"),
            "share_up": sum(x > 0 for x in deltas) / len(deltas) if deltas else float("nan"),
            "share_down": sum(x < 0 for x in deltas) / len(deltas) if deltas else float("nan"),
            "wilcoxon_p": pval,
        }
    return out


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "-" for c in s).strip("-")
