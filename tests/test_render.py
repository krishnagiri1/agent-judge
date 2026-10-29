from agent_judge.data import extract_issue
from agent_judge.render import render_transcript, truncate_middle

from .conftest import make_traj


def test_extract_issue_strips_boilerplate():
    text = "We're solving this.\nISSUE:\nCrash on empty input\nDetails here\n\nINSTRUCTIONS:\nNow, you're going to..."
    assert extract_issue(text) == "Crash on empty input\nDetails here"


def test_truncate_middle_keeps_head_and_tail():
    text = "HEAD" + "x" * 5000 + "TAIL"
    out = truncate_middle(text, 300)
    assert out.startswith("HEAD") and out.endswith("TAIL") and len(out) <= 300
    assert "truncated" in out


def test_render_respects_budget_and_marks_omissions():
    t = make_traj(1, resolved=True, n_steps=80)
    out = render_transcript(t, max_chars=8_000)
    assert len(out) <= 8_000
    assert "steps omitted" in out
    assert "## Step 1\n" in out  # head kept
    assert "submit" in out  # final steps kept
    assert "# FINAL PATCH" in out


def test_render_never_leaks_ground_truth():
    t = make_traj(2, resolved=True)
    out = render_transcript(t)
    assert "resolved" not in out.lower()
    assert "test-model" not in out


def test_filler_does_not_change_which_steps_are_kept():
    t = make_traj(3, resolved=False, n_steps=80)
    plain = render_transcript(t, max_chars=8_000)
    padded = render_transcript(t, max_chars=8_000, filler=lambda s: "FILLER sentence.")
    steps = lambda s: [line for line in s.splitlines() if line.startswith("## Step")]
    assert steps(plain) == steps(padded)
    assert padded.count("FILLER sentence.") == len(steps(plain))
