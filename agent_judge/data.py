"""Load and sample transcripts from the public nebius/SWE-agent-trajectories dataset.

The dataset (CC-BY-4.0) holds ~80k SWE-agent runs on real GitHub issues. Each run has a
`target` flag saying whether its final patch passed the repository's tests. We keep that
flag as hidden ground truth: it is never shown to the judge or to the human labeler.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

DATASET_ID = "nebius/SWE-agent-trajectories"
N_SHARDS = 12


@dataclass
class Trajectory:
    id: str
    instance_id: str
    model_name: str
    resolved: bool  # hidden ground truth: did the patch pass the tests?
    exit_status: str
    issue: str
    steps: list[dict]  # [{"role": "ai" | "user", "text": str}], system prompt and issue removed
    patch: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        return cls(**d)


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def shard_filename(i: int) -> str:
    return f"data/train-{i:05d}-of-{N_SHARDS:05d}.parquet"


def download_shards(n_shards: int, cache_dir: str | Path = "data/raw") -> list[Path]:
    """Download the first `n_shards` parquet shards (~90 MB each) from the Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    paths = []
    for i in range(min(n_shards, N_SHARDS)):
        p = hf_hub_download(
            repo_id=DATASET_ID,
            filename=shard_filename(i),
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
        paths.append(Path(p))
    return paths


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def extract_issue(first_user_text: str) -> str:
    """Pull the GitHub issue out of SWE-agent's first user message (drops the boilerplate)."""
    text = first_user_text or ""
    start = text.find("ISSUE:")
    start = start + len("ISSUE:") if start != -1 else 0
    end = text.find("\nINSTRUCTIONS:", start)
    if end == -1:
        end = len(text)
    return text[start:end].strip().replace("\r\n", "\n")


def row_to_trajectory(row: dict, uid: str) -> Trajectory:
    msgs = list(row["trajectory"])
    issue = ""
    steps: list[dict] = []
    issue_found = False
    for m in msgs:
        role = m.get("role")
        text = (m.get("text") or "").replace("\r\n", "\n")
        if role == "system":
            continue
        if role == "user" and not issue_found:
            issue = extract_issue(text)
            issue_found = True
            continue
        if role in ("ai", "user"):
            steps.append({"role": role, "text": text})
    return Trajectory(
        id=uid,
        instance_id=row["instance_id"],
        model_name=row.get("model_name") or "",
        resolved=bool(row["target"]),
        exit_status=row.get("exit_status") or "",
        issue=issue,
        steps=steps,
        patch=(row.get("generated_patch") or "").strip(),
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_trajectories(
    parquet_paths: Iterable[str | Path],
    n_instances: int,
    seed: int = 0,
) -> list[Trajectory]:
    """Pick `n_instances` issues that have both a resolved and an unresolved run.

    For each issue we keep one resolved and one unresolved trajectory, which gives
    (a) a 50/50 balanced pointwise set of 2 * n_instances transcripts and
    (b) natural pairs for the position-bias experiment.
    The returned list is shuffled so that the two runs of an issue are not adjacent
    (avoids anchoring when hand-labeling).
    """
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    candidates: list[tuple[Path, str, list[int], list[int]]] = []
    seen: set[str] = set()

    for path in parquet_paths:
        path = Path(path)
        meta = pq.read_table(path, columns=["instance_id", "target"]).to_pydict()
        by_inst: dict[str, tuple[list[int], list[int]]] = {}
        for idx, (iid, tgt) in enumerate(zip(meta["instance_id"], meta["target"])):
            pos, neg = by_inst.setdefault(iid, ([], []))
            (pos if tgt else neg).append(idx)
        for iid, (pos, neg) in by_inst.items():
            if pos and neg and iid not in seen:
                candidates.append((path, iid, pos, neg))
                seen.add(iid)
        del meta

    if len(candidates) < n_instances:
        raise ValueError(
            f"Only {len(candidates)} issues have both outcomes in these shards; "
            f"download more shards or lower --n-instances (asked for {n_instances})."
        )

    chosen = rng.sample(candidates, n_instances)
    picks_by_path: dict[Path, list[tuple[str, int, str]]] = {}
    for path, iid, pos, neg in chosen:
        picks_by_path.setdefault(path, []).append((iid, rng.choice(pos), "r"))
        picks_by_path[path].append((iid, rng.choice(neg), "u"))

    out: list[Trajectory] = []
    for path, picks in picks_by_path.items():
        table = pq.read_table(path)
        rows = table.take([idx for _, idx, _ in picks]).to_pylist()
        for (iid, idx, tag), row in zip(picks, rows):
            out.append(row_to_trajectory(row, uid=f"{iid}::{tag}{idx}"))
        del table

    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def save_jsonl(records: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(record: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_by_id(records: Iterable[dict]) -> dict[str, dict]:
    """Later records win, so re-labeling or re-judging an item simply overrides it."""
    out: dict[str, dict] = {}
    for r in records:
        out[r["id"]] = r
    return out


def save_trajectories(trajs: Iterable[Trajectory], path: str | Path) -> None:
    save_jsonl((t.to_dict() for t in trajs), path)


def load_trajectories(path: str | Path) -> list[Trajectory]:
    return [Trajectory.from_dict(d) for d in load_jsonl(path)]
