"""Build a separate imitation dataset from manually downloaded public games."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .outcome_logger import SUMMARY_FIELDS, summarize_trajectory
from .replay_parser import ReplayParseError, iter_transitions, load_replay


def _selected_seats(
    replay,
    *,
    winner_only: bool,
    minimum_reward: float,
) -> tuple[int, ...]:
    seats = []
    best_reward = max(replay.rewards)
    for seat, reward in enumerate(replay.rewards):
        if reward < minimum_reward:
            continue
        if winner_only and reward < best_reward:
            continue
        seats.append(seat)
    return tuple(seats)


def build_teacher_transitions(
    replay_directory: str | Path,
    output_directory: str | Path,
    *,
    winner_only: bool = False,
    minimum_reward: float = 0.0,
    minimum_steps: int = 700,
) -> dict[str, Any]:
    """Extract selected trajectories from both seats of foreign replays."""

    replay_directory = Path(replay_directory)
    output_directory = Path(output_directory)
    paths = sorted(replay_directory.glob("*.json"))
    if not paths:
        raise ValueError(f"No JSON replay files found in {replay_directory}")

    output_directory.mkdir(parents=True, exist_ok=True)
    transitions_path = output_directory / "transitions.jsonl.gz"
    episodes_path = output_directory / "episodes.csv"
    manifest_path = output_directory / "manifest.json"

    seen_episode_ids: set[int] = set()
    selected_summaries: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    skipped_files: list[dict[str, str]] = []
    transition_count = 0

    with gzip.open(transitions_path, "wt", encoding="utf-8", newline="\n") as output:
        for path in paths:
            try:
                replay = load_replay(path)
            except ReplayParseError as exc:
                skipped["invalid_replay"] += 1
                skipped_files.append({"file": path.name, "reason": str(exc)})
                continue

            if replay.episode_id in seen_episode_ids:
                skipped["duplicate_episode"] += 1
                skipped_files.append({"file": path.name, "reason": "duplicate_episode"})
                continue
            seen_episode_ids.add(replay.episode_id)

            if any(status.upper() != "DONE" for status in replay.statuses):
                skipped["unfinished"] += 1
                skipped_files.append({"file": path.name, "reason": "unfinished"})
                continue
            if len(replay.steps) - 1 < minimum_steps:
                skipped["too_short"] += 1
                skipped_files.append({"file": path.name, "reason": "too_short"})
                continue

            seats = _selected_seats(
                replay,
                winner_only=winner_only,
                minimum_reward=minimum_reward,
            )
            if not seats:
                skipped["no_selected_trajectory"] += 1
                skipped_files.append(
                    {"file": path.name, "reason": "no_selected_trajectory"}
                )
                continue

            selected = set(seats)
            selected_summaries.extend(
                summarize_trajectory(replay, seat) for seat in seats
            )
            for transition in iter_transitions(replay, team_name=None):
                if transition.seat not in selected:
                    continue
                record = transition.as_dict()
                record["source_type"] = "teacher_replay"
                record["teacher_winner"] = transition.outcome == "win"
                json.dump(record, output, ensure_ascii=True, separators=(",", ":"))
                output.write("\n")
                transition_count += 1

    with episodes_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(selected_summaries)

    teams = Counter(str(row["team"]) for row in selected_summaries)
    outcomes = Counter(str(row["result"]) for row in selected_summaries)
    rewards = [float(row["reward"]) for row in selected_summaries]
    manifest = {
        "version": 1,
        "source_type": "teacher_replay",
        "source_directory": str(replay_directory.resolve()),
        "json_files_found": len(paths),
        "unique_episodes_seen": len(seen_episode_ids),
        "selected_trajectories": len(selected_summaries),
        "transitions": transition_count,
        "filters": {
            "winner_only": winner_only,
            "minimum_reward": minimum_reward,
            "minimum_steps": minimum_steps,
            "require_done_status": True,
        },
        "outcomes": dict(sorted(outcomes.items())),
        "teams": dict(sorted(teams.items())),
        "reward": {
            "minimum": min(rewards) if rewards else None,
            "maximum": max(rewards) if rewards else None,
        },
        "skipped": dict(sorted(skipped.items())),
        "skipped_files": skipped_files,
        "files": {
            "transitions": transitions_path.name,
            "episodes": episodes_path.name,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return manifest
