"""Agreement metrics between the LLM judge, the human labels, and test-based ground truth."""

from __future__ import annotations

import math
import random
from typing import Optional

from .rubric import DIMENSIONS


def _clean(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def weighted_kappa(a: list[int], b: list[int]) -> float:
    """Quadratic-weighted Cohen's kappa on the 1-5 scale (standard for ordinal ratings)."""
    from sklearn.metrics import cohen_kappa_score

    a, b = _clean(a, b)
    if len(a) < 2:
        return float("nan")
    if len(set(a)) == 1 and len(set(b)) == 1:
        return 1.0 if a[0] == b[0] else 0.0
    return float(cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5]))


def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    a, b = _clean(a, b)
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def auroc(scores, labels) -> float:
    from sklearn.metrics import roc_auc_score

    scores, labels = _clean(scores, labels)
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score([int(bool(x)) for x in labels], scores))


def bootstrap_ci(fn, a, b, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    a, b = _clean(a, b)
    if len(a) < 5:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    idx = list(range(len(a)))
    vals = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        v = fn([a[i] for i in s], [b[i] for i in s])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    lo = vals[int(alpha / 2 * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return (lo, hi)


def agreement_table(human: dict[str, dict], judge: dict[str, dict]) -> dict:
    """Per-dimension judge-vs-human agreement over the items both have scored."""
    ids = [i for i in human if i in judge and judge[i].get("scores") and human[i].get("scores")]
    out: dict = {"n": len(ids), "dims": {}}
    for d in DIMENSIONS:
        h = [human[i]["scores"].get(d) for i in ids]
        j = [judge[i]["scores"].get(d) for i in ids]
        hc, jc = _clean(h, j)
        k = weighted_kappa(hc, jc)
        out["dims"][d] = {
            "kappa": k,
            "kappa_ci95": bootstrap_ci(weighted_kappa, hc, jc),
            "spearman": spearman(hc, jc),
            "exact": sum(x == y for x, y in zip(hc, jc)) / len(hc) if hc else float("nan"),
            "within_1": sum(abs(x - y) <= 1 for x, y in zip(hc, jc)) / len(hc) if hc else float("nan"),
            "mean_human": sum(hc) / len(hc) if hc else float("nan"),
            "mean_judge": sum(jc) / len(jc) if jc else float("nan"),
        }
    return out


def ground_truth_table(records: dict[str, dict], truth: dict[str, bool]) -> dict:
    """How well a rater's scores separate test-passing from test-failing runs (AUROC)."""
    ids = [i for i in records if i in truth and records[i].get("scores")]
    y = [truth[i] for i in ids]
    out = {"n": len(ids), "base_rate": (sum(y) / len(y)) if y else float("nan")}
    probs = [records[i].get("resolved_probability") for i in ids]
    out["auroc_resolved_probability"] = auroc(probs, y)
    out["auroc_resolved_probability_ci95"] = bootstrap_ci(auroc, probs, y)
    for d in DIMENSIONS:
        out[f"auroc_{d}"] = auroc([records[i]["scores"].get(d) for i in ids], y)
    return out


def fmt(x: Optional[float], digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.0f}%"
