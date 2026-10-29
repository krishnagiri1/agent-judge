import random

import pytest

from agent_judge.data import Trajectory


def make_traj(i: int, resolved: bool, n_steps: int = 12, wordy: bool = False) -> Trajectory:
    rng = random.Random(i)
    steps = []
    for s in range(n_steps):
        thought = f"Step {s}: I will inspect the module and run the reproduction script."
        if wordy:
            thought = ("I think carefully about this. " * 30) + thought
        steps.append({"role": "ai", "text": f"{thought}\n```\nopen src/mod_{s}.py\n```"})
        out = "ok" if resolved or rng.random() > 0.5 else "Traceback (most recent call last):\nSyntaxError: bad"
        steps.append({"role": "user", "text": f"[File: src/mod_{s}.py]\n{out}\n" + "x = 1\n" * rng.randint(1, 50)})
    if resolved:
        steps.append({"role": "ai", "text": "The fix is verified. Submitting.\n```\nsubmit\n```"})
    patch = "diff --git a/src/mod.py b/src/mod.py\n+    return value or default\n" if resolved else ""
    return Trajectory(
        id=f"repo__issue-{i // 2}::{'r' if resolved else 'u'}{i}",
        instance_id=f"repo__issue-{i // 2}",
        model_name="test-model",
        resolved=resolved,
        exit_status="submitted" if resolved else "submitted (exit_context)",
        issue=f"Function returns None instead of default (issue {i // 2}).",
        steps=steps,
        patch=patch,
    )


@pytest.fixture
def trajs():
    return [make_traj(i, resolved=(i % 2 == 0)) for i in range(20)]
