import math

import pytest

from agent_judge.judge import ParseError, parse_pairwise, parse_pointwise
from agent_judge.metrics import agreement_table, auroc, weighted_kappa


def test_parse_pointwise_handles_fences_and_scales():
    raw = 'Sure!\n```json\n{"analysis": "x", "correctness": 4.4, "tool_use": 9, "reasoning": "3", ' \
          '"overall": 4, "resolved_probability": 0.8, "rationale": "ok"}\n```'
    p = parse_pointwise(raw)
    assert p["scores"] == {"correctness": 4, "tool_use": 5, "reasoning": 3, "overall": 4}
    assert p["resolved_probability"] == pytest.approx(80)


def test_parse_pointwise_rejects_missing_fields():
    with pytest.raises(ParseError):
        parse_pointwise('{"correctness": 3}')
    with pytest.raises(ParseError):
        parse_pointwise("no json here")


def test_parse_pairwise():
    assert parse_pairwise('{"winner": "b"}')["winner"] == "B"
    assert parse_pairwise('{"winner": "Tie"}')["winner"] == "tie"
    with pytest.raises(ParseError):
        parse_pairwise('{"winner": "C"}')


def test_kappa_perfect_and_chance():
    a = [1, 2, 3, 4, 5, 3, 2]
    assert weighted_kappa(a, a) == pytest.approx(1.0)
    assert weighted_kappa(a, [6 - x for x in a]) < 0


def test_auroc():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert math.isnan(auroc([0.1, 0.2], [True, True]))


def test_agreement_table_aligns_ids():
    human = {"a": {"scores": dict(correctness=5, tool_use=4, reasoning=4, overall=5)},
             "b": {"scores": dict(correctness=1, tool_use=2, reasoning=1, overall=1)},
             "c": {"scores": dict(correctness=3, tool_use=3, reasoning=3, overall=3)}}
    judge = {k: v for k, v in human.items() if k != "c"} | {"z": human["a"]}
    t = agreement_table(human, judge)
    assert t["n"] == 2
    assert t["dims"]["overall"]["exact"] == 1.0
