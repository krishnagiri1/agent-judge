"""Turn a Trajectory into a bounded-length text transcript.

Raw SWE-agent runs average 8k-15k tokens and some exceed 30k, so we:
  * cap every agent turn and every observation, keeping the head AND tail
    (actions sit at the end of agent turns, error messages at the end of outputs);
  * if still over budget, keep the first `head_steps` steps and as many final
    steps as fit, replacing the middle with an explicit "[... N steps omitted ...]".

The judge and the human labeler see exactly the same rendered text, so agreement
numbers are not confounded by one side seeing more of the run than the other.
The hidden `resolved` label and the evaluation logs are never rendered.
"""

from __future__ import annotations

from typing import Callable, Optional

from .data import Trajectory

DEFAULT_RENDER = {
    "max_chars": 28_000,  # ~7k tokens; fits local 16k-context models with room for the rubric
    "max_ai_chars": 1_500,
    "max_obs_chars": 1_200,
    "max_issue_chars": 3_000,
    "max_patch_chars": 4_000,
    "head_steps": 8,
}


def truncate_middle(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    marker = f"\n[... {len(text) - limit} chars truncated ...]\n"
    keep = max(limit - len(marker), 0)
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def _build_steps(traj: Trajectory, max_ai_chars: int, max_obs_chars: int) -> list[tuple[str, str]]:
    """Group the flat message list into (agent_turn, observation) steps."""
    steps: list[tuple[str, str]] = []
    current_ai: Optional[str] = None
    for m in traj.steps:
        if m["role"] == "ai":
            if current_ai is not None:
                steps.append((current_ai, ""))
            current_ai = truncate_middle(m["text"], max_ai_chars)
        else:  # observation from the environment
            obs = truncate_middle(m["text"], max_obs_chars) or "(no output)"
            if current_ai is None:
                steps.append(("", obs))
            else:
                steps.append((current_ai, obs))
                current_ai = None
    if current_ai is not None:
        steps.append((current_ai, ""))
    return steps


def _step_block(i: int, ai: str, obs: str, filler: str = "") -> str:
    parts = [f"## Step {i}"]
    if ai or filler:
        body = f"{filler}\n\n{ai}".strip() if filler else ai
        parts.append(f"AGENT:\n{body}")
    if obs:
        parts.append(f"OBSERVATION:\n{obs}")
    return "\n".join(parts)


def render_transcript(
    traj: Trajectory,
    max_chars: int = DEFAULT_RENDER["max_chars"],
    max_ai_chars: int = DEFAULT_RENDER["max_ai_chars"],
    max_obs_chars: int = DEFAULT_RENDER["max_obs_chars"],
    max_issue_chars: int = DEFAULT_RENDER["max_issue_chars"],
    max_patch_chars: int = DEFAULT_RENDER["max_patch_chars"],
    head_steps: int = DEFAULT_RENDER["head_steps"],
    filler: Optional[Callable[[int], str]] = None,
) -> str:
    """Render a transcript for judging.

    `filler(step_number) -> str` injects extra text at the start of each agent turn.
    It is used by the verbosity-bias test and is deliberately excluded from length
    budgeting and truncation, so the padded and original renders keep exactly the
    same steps and differ only by the filler.
    """
    issue = truncate_middle(traj.issue, max_issue_chars)
    patch = truncate_middle(traj.patch, max_patch_chars) if traj.patch else "(no patch was produced)"
    steps = _build_steps(traj, max_ai_chars, max_obs_chars)
    n = len(steps)

    plain_blocks = [_step_block(i + 1, ai, obs) for i, (ai, obs) in enumerate(steps)]

    header = f"# ISSUE\n{issue}\n\n# AGENT TRAJECTORY ({n} steps, exit status: {traj.exit_status or 'unknown'})"
    footer = f"# FINAL PATCH\n```diff\n{patch}\n```"
    budget = max_chars - len(header) - len(footer) - 200

    # Choose which steps to keep based on the unpadded text only.
    keep = list(range(n))
    if sum(len(b) + 2 for b in plain_blocks) > budget:
        head = list(range(min(head_steps, n)))
        used = sum(len(plain_blocks[i]) + 2 for i in head)
        tail: list[int] = []
        for i in range(n - 1, len(head) - 1, -1):
            cost = len(plain_blocks[i]) + 2
            if used + cost > budget:
                break
            tail.append(i)
            used += cost
        keep = head + sorted(tail)

    out: list[str] = []
    prev = -1
    for i in keep:
        if i != prev + 1:
            out.append(f"[... {i - prev - 1} steps omitted ...]")
        ai, obs = steps[i]
        out.append(_step_block(i + 1, ai, obs, filler(i + 1) if filler else ""))
        prev = i
    if prev != n - 1:
        out.append(f"[... {n - 1 - prev} steps omitted ...]")

    return header + "\n\n" + "\n\n".join(out) + "\n\n" + footer
