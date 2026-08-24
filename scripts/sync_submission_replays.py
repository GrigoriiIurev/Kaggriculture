"""Incrementally synchronize every completed replay for one submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.update_replays import (
    _completed_episode_ids,
    _existing_episode_ids,
    _find_kaggle,
    _parse_episode_output,
    _run_kaggle,
)
from src.kaggriculture.data.replay_parser import load_replay


DEFAULT_COMPETITION = "kaggriculture"


def _parse_json_list(output: str, label: str) -> list[dict[str, Any]]:
    try:
        value, _ = json.JSONDecoder().raw_decode(output.lstrip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Kaggle CLI did not return a JSON {label} list") from exc
    if not isinstance(value, list):
        raise RuntimeError(f"Kaggle CLI {label} response is not a list")
    return [item for item in value if isinstance(item, dict)]


def select_latest_completed_submission(
    submissions: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = [
        item
        for item in submissions
        if str(item.get("status", "")).upper().endswith("COMPLETE")
        and item.get("ref") is not None
    ]
    if not completed:
        raise RuntimeError("No completed Kaggle submission was found")
    return max(completed, key=lambda item: str(item.get("date", "")))


def sync_submission_replays(
    replay_directory: str | Path,
    index_path: str | Path,
    submission_id: int | None = None,
    competition: str = DEFAULT_COMPETITION,
    kaggle_path: str | None = None,
    max_replays: int = 0,
) -> dict[str, Any]:
    executable, environment = _find_kaggle(kaggle_path)
    submissions = _parse_json_list(
        _run_kaggle(
            executable,
            environment,
            ["competitions", "submissions", competition, "--format", "json"],
        ),
        "submission",
    )
    if submission_id is None:
        submission = select_latest_completed_submission(submissions)
        submission_id = int(submission["ref"])
    else:
        submission = next(
            (item for item in submissions if int(item.get("ref", -1)) == submission_id),
            {"ref": submission_id},
        )
    print(
        f"[sync] submission {submission_id}: "
        f"{submission.get('description', '')} "
        f"(score={submission.get('publicScore', '')})",
        flush=True,
    )

    episode_output = _run_kaggle(
        executable,
        environment,
        [
            "competitions",
            "episodes",
            str(submission_id),
            "--format",
            "json",
        ],
    )
    episodes = _parse_episode_output(episode_output)
    completed_ids = set(_completed_episode_ids(episodes))
    selected_episodes = [
        item
        for item in episodes
        if int(item.get("id", -1)) in completed_ids
    ]
    selected_episodes.sort(
        key=lambda item: (str(item.get("createTime", "")), int(item["id"])),
        reverse=True,
    )
    if max_replays > 0:
        selected_episodes = selected_episodes[:max_replays]
    selected_ids = {int(item["id"]) for item in selected_episodes}

    replay_directory = Path(replay_directory)
    index_path = Path(index_path)
    replay_directory.mkdir(parents=True, exist_ok=True)
    existing = _existing_episode_ids(replay_directory)
    missing = sorted(selected_ids - existing)
    print(
        f"[sync] {len(selected_ids):,} completed episodes selected; "
        f"{len(existing & selected_ids):,} already present; "
        f"{len(missing):,} to download",
        flush=True,
    )
    for number, episode_id in enumerate(missing, start=1):
        print(
            f"[download {number:,}/{len(missing):,}] episode {episode_id}",
            flush=True,
        )
        _run_kaggle(
            executable,
            environment,
            [
                "competitions",
                "replay",
                str(episode_id),
                "--path",
                str(replay_directory),
                "--quiet",
            ],
            timeout=300,
        )
        matches = list(replay_directory.glob(f"*{episode_id}*.json"))
        if len(matches) != 1:
            raise RuntimeError(
                f"Episode {episode_id} download did not produce exactly one replay"
            )
        replay = load_replay(matches[0])
        if replay.episode_id != episode_id:
            raise RuntimeError(
                f"Downloaded episode id mismatch: {replay.episode_id} != {episode_id}"
            )

    payload = {
        "version": 1,
        "competition": competition,
        "submission": submission,
        "episodes": selected_episodes,
        "remote_completed": len(completed_ids),
        "selected": len(selected_ids),
        "downloaded_now": len(missing),
        "local_selected": len(_existing_episode_ids(replay_directory) & selected_ids),
        "replay_directory": str(replay_directory.resolve()),
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_name(f".{index_path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(index_path)
    print(f"[sync] index saved to {index_path}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-id", type=int)
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--kaggle")
    parser.add_argument("--max-replays", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = sync_submission_replays(
        replay_directory=args.replays,
        index_path=args.index,
        submission_id=args.submission_id,
        competition=args.competition,
        kaggle_path=args.kaggle,
        max_replays=args.max_replays,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
