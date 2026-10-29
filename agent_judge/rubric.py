"""The scoring rubric shared by the LLM judge and the human labeling app.

Keeping one definition for both is what makes judge-human agreement meaningful:
the human and the model score the same transcript against the same anchors.
"""

from __future__ import annotations

SCORE_DIMENSIONS: dict[str, dict] = {
    "correctness": {
        "question": "Does the final patch fix the root cause described in the issue without breaking anything else?",
        "anchors": {
            1: "No patch, an empty or irrelevant patch, or a change that clearly cannot fix the issue.",
            2: "Touches the right area but the fix is wrong, partial, or likely introduces new bugs.",
            3: "Plausible fix with real doubts: misses an edge case, or the root cause is only partly addressed.",
            4: "Likely correct fix of the root cause; minor concerns (style, leftover debug code, extra files).",
            5: "Clean, minimal fix that clearly addresses the root cause; nothing unrelated changed.",
        },
    },
    "tool_use": {
        "question": "Did the agent use its tools effectively: right commands, reading outputs, no thrashing?",
        "anchors": {
            1: "Repeated failing commands, loops, or ignored errors; ran out of steps or context.",
            2: "Many wasted or malformed actions (e.g. repeated broken edits) before making progress.",
            3: "Gets there, but with noticeable inefficiency or a few avoidable mistakes.",
            4: "Mostly efficient: targeted searches and edits, reacts correctly to tool output.",
            5: "Efficient and deliberate throughout: finds the right files fast, edits cleanly, verifies.",
        },
    },
    "reasoning": {
        "question": "Is the agent's reasoning grounded in evidence: reproduce, diagnose, fix, verify?",
        "anchors": {
            1: "Guesses or contradicts the evidence; no reproduction or verification.",
            2: "Some diagnosis, but conclusions are weakly supported or the verification is skipped.",
            3: "Reasonable diagnosis; verification is shallow or the agent misreads some output.",
            4: "Clear hypothesis grounded in observations; reproduces the bug and verifies the fix.",
            5: "Rigorous: reproduces, isolates the root cause, fixes it, and confirms with evidence.",
        },
    },
    "overall": {
        "question": "Overall, how good is this transcript as an example of solving the task?",
        "anchors": {
            1: "Failed attempt with poor process.",
            2: "Mostly failed; a few useful steps.",
            3: "Mixed: partial success or a correct-looking result reached sloppily.",
            4: "Good attempt with minor flaws.",
            5: "Excellent attempt you would use as a reference example.",
        },
    },
}

DIMENSIONS: list[str] = list(SCORE_DIMENSIONS)

RESOLVED_QUESTION = (
    "Estimate the probability (0-100) that the final patch passes the repository's hidden tests "
    "for this issue."
)


def rubric_text() -> str:
    """Render the rubric as plain text for prompts and the labeling guide."""
    lines = []
    for dim, spec in SCORE_DIMENSIONS.items():
        lines.append(f"{dim.upper()}: {spec['question']}")
        for score, desc in spec["anchors"].items():
            lines.append(f"  {score} = {desc}")
        lines.append("")
    lines.append(f"RESOLVED_PROBABILITY: {RESOLVED_QUESTION}")
    return "\n".join(lines)
