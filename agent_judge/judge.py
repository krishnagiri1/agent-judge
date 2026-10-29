"""Pointwise and pairwise LLM judging of agent transcripts."""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

from .data import Trajectory, append_jsonl, load_jsonl
from .llm import LLMClient
from .render import render_transcript
from .rubric import DIMENSIONS, rubric_text

SYSTEM_PROMPT = (
    "You are a senior software engineer who reviews transcripts of an autonomous coding agent "
    "working on real GitHub issues. You grade carefully and consistently against the rubric you "
    "are given, and you answer with a single JSON object and nothing else."
)

_SCORE_SCHEMA = (
    '"correctness": <1-5>, "tool_use": <1-5>, "reasoning": <1-5>, "overall": <1-5>, '
    '"resolved_probability": <0-100>, "rationale": "<2-3 sentences citing specific steps>"'
)

POINTWISE_PROMPTS: dict[str, str] = {
    # v1: score directly.
    "v1": (
        "Grade the coding-agent transcript below using this rubric.\n\n"
        "RUBRIC\n{rubric}\n\n"
        "{transcript}\n\n"
        "Return only a JSON object of the form:\n{{" + _SCORE_SCHEMA + "}}"
    ),
    # v2: analyze first against explicit checks, and judge substance rather than style.
    "v2": (
        "Grade the coding-agent transcript below using this rubric.\n\n"
        "RUBRIC\n{rubric}\n\n"
        "Before scoring, work through these checks in an \"analysis\" field:\n"
        "1. What is the root cause of the issue, in one sentence?\n"
        "2. Did the agent reproduce the bug? Cite the step.\n"
        "3. Does the final patch change the code responsible for the root cause? Is anything unrelated changed "
        "(debug prints, reproduction scripts, unrelated files)?\n"
        "4. Did the agent verify the fix by re-running the reproduction or tests? Cite the step.\n"
        "5. Were there loops, repeated failing edits, or ignored errors?\n"
        "Judge substance, not style: long, confident, or polite narration is not evidence of good work. "
        "Only actions, tool outputs, and the final patch count.\n\n"
        "{transcript}\n\n"
        "Return only a JSON object of the form:\n"
        '{{"analysis": "<your checks 1-5>", ' + _SCORE_SCHEMA + "}}"
    ),
}

PAIRWISE_PROMPTS: dict[str, str] = {
    "baseline": (
        "Two attempts by a coding agent at the same GitHub issue are shown below. "
        "Decide which attempt better solves the issue.\n\n"
        "===== TRANSCRIPT A =====\n{a}\n\n===== TRANSCRIPT B =====\n{b}\n\n"
        'Return only a JSON object: {{"analysis": "<brief comparison>", "winner": "A" | "B" | "tie", '
        '"confidence": <0-100>}}'
    ),
    "debiased": (
        "Two attempts by a coding agent at the same GitHub issue are shown below. Their order is random "
        "and carries no information. Length and tone are not evidence; only actions, tool outputs, and "
        "the final patch count.\n\n"
        "===== TRANSCRIPT A =====\n{a}\n\n===== TRANSCRIPT B =====\n{b}\n\n"
        "First assess each attempt on its own: does its final patch fix the root cause, and did the agent "
        "verify it? Only then compare.\n"
        'Return only a JSON object: {{"assessment_a": "<on its own>", "assessment_b": "<on its own>", '
        '"winner": "A" | "B" | "tie", "confidence": <0-100>}}'
    ),
}

RETRY_SUFFIX = (
    "\n\nIMPORTANT: your previous reply could not be parsed ({error}). "
    "Reply with exactly one valid JSON object with all required fields and no other text."
)


class ParseError(ValueError):
    pass


def extract_json(text: str) -> dict:
    """Find and parse the JSON object in a model reply (tolerates code fences and preambles)."""
    if not text:
        raise ParseError("empty reply")
    cleaned = re.sub(r"```(?:json)?", "", text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ParseError("no JSON object found")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e.msg}") from e


def _to_score(v) -> int:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ParseError(f"non-numeric score {v!r}")
    return int(min(5, max(1, round(x))))


def parse_pointwise(text: str) -> dict:
    d = extract_json(text)
    missing = [k for k in DIMENSIONS + ["resolved_probability"] if k not in d]
    if missing:
        raise ParseError(f"missing fields {missing}")
    scores = {k: _to_score(d[k]) for k in DIMENSIONS}
    try:
        p = float(d["resolved_probability"])
    except (TypeError, ValueError):
        raise ParseError("non-numeric resolved_probability")
    if 0 < p <= 1 and isinstance(d["resolved_probability"], float):
        p *= 100  # model answered on a 0-1 scale
    return {
        "scores": scores,
        "resolved_probability": float(min(100, max(0, p))),
        "rationale": str(d.get("rationale", "")),
        "analysis": str(d.get("analysis", "")),
    }


def parse_pairwise(text: str) -> dict:
    d = extract_json(text)
    w = str(d.get("winner", "")).strip().upper()
    if w in ("TIE", "EQUAL", "NEITHER", "BOTH"):
        w = "tie"
    if w not in ("A", "B", "tie"):
        raise ParseError(f"bad winner {d.get('winner')!r}")
    return {"winner": w, "confidence": d.get("confidence"), "analysis": d.get("analysis") or d}


def _call_with_parse(client: LLMClient, user: str, parser, retries: int = 2):
    prompt = user
    err = None
    raw = ""
    for _ in range(retries + 1):
        raw = client.complete(SYSTEM_PROMPT, prompt)
        try:
            return parser(raw), raw, None
        except ParseError as e:
            err = str(e)
            prompt = user + RETRY_SUFFIX.format(error=err)
    return None, raw, err


def build_pointwise_prompt(traj: Trajectory, prompt_version: str = "v2", render_kwargs: Optional[dict] = None,
                           filler=None) -> str:
    if prompt_version not in POINTWISE_PROMPTS:
        raise ValueError(f"Unknown prompt version {prompt_version!r}; choose from {list(POINTWISE_PROMPTS)}")
    transcript = render_transcript(traj, filler=filler, **(render_kwargs or {}))
    return POINTWISE_PROMPTS[prompt_version].format(rubric=rubric_text(), transcript=transcript)


def judge_one(client: LLMClient, traj: Trajectory, prompt_version: str = "v2",
              render_kwargs: Optional[dict] = None, filler=None) -> dict:
    t0 = time.time()
    user = build_pointwise_prompt(traj, prompt_version, render_kwargs, filler)
    parsed, raw, err = _call_with_parse(client, user, parse_pointwise)
    rec = {
        "id": traj.id,
        "judge": client.name,
        "prompt": prompt_version,
        "error": err,
        "raw": raw[:4000],
        "elapsed_s": round(time.time() - t0, 2),
    }
    if parsed:
        rec.update(parsed)
    return rec


def judge_many(
    client: LLMClient,
    trajs: Iterable[Trajectory],
    prompt_version: str,
    out_path: str | Path,
    workers: int = 4,
    render_kwargs: Optional[dict] = None,
    filler=None,
    progress: bool = True,
) -> list[dict]:
    """Judge many transcripts in parallel, appending to `out_path` as results arrive.

    Re-running is safe: transcripts that already have a successful judgment in
    `out_path` are skipped, so an interrupted run picks up where it left off.
    """
    trajs = list(trajs)
    done = {r["id"]: r for r in load_jsonl(out_path) if not r.get("error")}
    todo = [t for t in trajs if t.id not in done]
    lock = threading.Lock()
    results = dict(done)

    bar = None
    if progress and todo:
        try:
            from tqdm import tqdm

            bar = tqdm(total=len(todo), desc=f"judging ({client.name}, {prompt_version})")
        except ImportError:
            pass

    def work(t: Trajectory):
        try:
            return judge_one(client, t, prompt_version, render_kwargs, filler)
        except Exception as e:  # network failure after retries etc.
            return {"id": t.id, "judge": client.name, "prompt": prompt_version, "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(work, t) for t in todo]
        for f in as_completed(futures):
            rec = f.result()
            with lock:
                append_jsonl(rec, out_path)
                results[rec["id"]] = rec
                if bar:
                    bar.update(1)
    if bar:
        bar.close()
    return [results[t.id] for t in trajs if t.id in results]


def judge_pair(client: LLMClient, a: Trajectory, b: Trajectory, prompt: str = "baseline",
               render_kwargs: Optional[dict] = None) -> dict:
    """Ask which of two transcripts (same issue) is better, with `a` shown first."""
    if prompt not in PAIRWISE_PROMPTS:
        raise ValueError(f"Unknown pairwise prompt {prompt!r}; choose from {list(PAIRWISE_PROMPTS)}")
    rk = {"max_chars": 14_000, **(render_kwargs or {})}
    user = PAIRWISE_PROMPTS[prompt].format(a=render_transcript(a, **rk), b=render_transcript(b, **rk))
    parsed, raw, err = _call_with_parse(client, user, parse_pairwise)
    return {"first": a.id, "second": b.id, "error": err, "raw": raw[:2000], **(parsed or {})}
