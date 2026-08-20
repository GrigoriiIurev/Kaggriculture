"""Download available public replays from top leaderboard teams."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scripts.update_replays import _existing_episode_ids, _find_kaggle, _run_kaggle


DEFAULT_COMPETITION = "kaggriculture"


def _parse_json_list(output: str, description: str) -> list[dict[str, Any]]:
    """Read a JSON list even when Kaggle prints a page-token line first."""

    start = output.find("[")
    if start < 0:
        raise RuntimeError(f"Kaggle CLI did not return {description} as JSON")
    try:
        value, _ = json.JSONDecoder().raw_decode(output[start:])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Kaggle CLI returned invalid {description} JSON") from exc
    if not isinstance(value, list):
        raise RuntimeError(f"Kaggle CLI {description} response is not a list")
    return [item for item in value if isinstance(item, dict)]


def _next_page_token(output: str) -> str | None:
    match = re.search(r"^Next Page Token = (\S+)\s*$", output, re.MULTILINE)
    return match.group(1) if match else None


def _score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _best_submission(submissions: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [item for item in submissions if item.get("id") is not None]
    if not valid:
        return None
    return max(
        valid,
        key=lambda item: (_score(item.get("publicScore")), str(item.get("dateSubmitted", ""))),
    )


def _completed_public_episode_ids(
    episodes: list[dict[str, Any]], limit: int | None = None
) -> list[int]:
    completed: dict[int, str] = {}
    for episode in episodes:
        state = str(episode.get("state", "")).upper()
        episode_type = str(episode.get("type", "")).upper()
        if not state.endswith("COMPLETED") or not episode_type.endswith("PUBLIC"):
            continue
        try:
            episode_id = int(episode["id"])
        except (KeyError, TypeError, ValueError):
            continue
        created = str(episode.get("createTime", ""))
        completed[episode_id] = max(created, completed.get(episode_id, ""))

    ids = sorted(completed, key=lambda episode_id: (completed[episode_id], episode_id), reverse=True)
    return ids[:limit] if limit is not None else ids


def _leaderboard(
    executable: Path,
    environment: dict[str, str],
    competition: str,
    count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_token: str | None = None
    while len(rows) < count:
        arguments = [
            "competitions",
            "leaderboard",
            competition,
            "--show",
            "--page-size",
            str(min(100, count - len(rows))),
            "--format",
            "json",
            "--quiet",
        ]
        if page_token:
            arguments.extend(["--page-token", page_token])
        output = _run_kaggle(executable, environment, arguments)
        page = _parse_json_list(output, "leaderboard")
        rows.extend(page)
        page_token = _next_page_token(output)
        if not page_token or not page:
            break
    return rows[:count]


def download_top_replays(
    top_players: int,
    replay_directory: Path,
    competition: str = DEFAULT_COMPETITION,
    max_replays_per_player: int | None = None,
    kaggle_path: str | None = None,
    list_only: bool = False,
    best_submission_only: bool = False,
) -> dict[str, Any]:
    if top_players < 1:
        raise ValueError("top_players must be at least 1")
    if max_replays_per_player is not None and max_replays_per_player < 1:
        raise ValueError("max_replays_per_player must be at least 1")

    executable, environment = _find_kaggle(kaggle_path)
    leaderboard = _leaderboard(
        executable, environment, competition=competition, count=top_players
    )

    players: list[dict[str, Any]] = []
    all_episode_ids: set[int] = set()
    for rank, row in enumerate(leaderboard, start=1):
        team_id = int(row["teamId"])
        submissions_output = _run_kaggle(
            executable,
            environment,
            ["competitions", "team-submissions", str(team_id), "--format", "json", "--quiet"],
        )
        submissions = _parse_json_list(
            submissions_output, f"submissions for team {team_id}"
        )
        submissions = [item for item in submissions if item.get("id") is not None]
        if best_submission_only:
            best = _best_submission(submissions)
            submissions = [best] if best is not None else []
        if not submissions:
            players.append(
                {
                    "rank": rank,
                    "team": row.get("teamName"),
                    "team_id": team_id,
                    "status": "no public submission",
                }
            )
            continue

        player_episodes: list[dict[str, Any]] = []
        submission_ids: list[int] = []
        for submission in submissions:
            submission_id = int(submission["id"])
            submission_ids.append(submission_id)
            episodes_output = _run_kaggle(
                executable,
                environment,
                [
                    "competitions",
                    "episodes",
                    str(submission_id),
                    "--format",
                    "json",
                    "--quiet",
                ],
            )
            player_episodes.extend(
                _parse_json_list(
                    episodes_output, f"episodes for submission {submission_id}"
                )
            )
        episode_ids = _completed_public_episode_ids(
            player_episodes,
            limit=max_replays_per_player,
        )
        all_episode_ids.update(episode_ids)
        players.append(
            {
                "rank": rank,
                "team": row.get("teamName"),
                "team_id": team_id,
                "leaderboard_score": row.get("score"),
                "submission_ids": submission_ids,
                "public_replays": len(episode_ids),
            }
        )

    existing_ids = _existing_episode_ids(replay_directory)
    missing_ids = sorted(all_episode_ids - existing_ids)
    result: dict[str, Any] = {
        "competition": competition,
        "requested_top_players": top_players,
        "players_found": len(leaderboard),
        "players": players,
        "unique_public_replays": len(all_episode_ids),
        "already_downloaded": len(all_episode_ids & existing_ids),
        "to_download": len(missing_ids),
    }
    if list_only:
        result["downloaded"] = 0
        return result

    replay_directory.mkdir(parents=True, exist_ok=True)
    for number, episode_id in enumerate(missing_ids, start=1):
        print(f"Downloading replay {number}/{len(missing_ids)}: {episode_id}", flush=True)
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
        )
    result["downloaded"] = len(missing_ids)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-players",
        type=int,
        required=True,
        help="Number of players to take from the top of the leaderboard",
    )
    parser.add_argument(
        "--max-replays-per-player",
        type=int,
        help="Download only this many newest replays per player (default: all)",
    )
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--replays", type=Path, default=Path("data/teacher_replays"))
    parser.add_argument("--kaggle", help="Path to the Kaggle executable")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Show what would be downloaded without downloading files",
    )
    parser.add_argument(
        "--best-submission-only",
        action="store_true",
        help="Use only each player's highest-scoring active submission",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = download_top_replays(
        top_players=args.top_players,
        replay_directory=args.replays,
        competition=args.competition,
        max_replays_per_player=args.max_replays_per_player,
        kaggle_path=args.kaggle,
        list_only=args.list_only,
        best_submission_only=args.best_submission_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
