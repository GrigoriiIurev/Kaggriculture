"""Download missing Kaggriculture replays and rebuild the local datasets."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from src.kaggriculture.data.pipeline import build_all_datasets


DEFAULT_SUBMISSION_ID = 55562698
DEFAULT_TEAM = "Grigorii IU"


def _find_kaggle(explicit_path: str | None) -> tuple[Path, dict[str, str]]:
    candidates = [
        Path(explicit_path).expanduser() if explicit_path else None,
        Path(".venv/bin/kaggle"),
        Path(shutil.which("kaggle")) if shutil.which("kaggle") else None,
        Path("/tmp/kaggle-cli/bin/kaggle"),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            environment = os.environ.copy()
            if candidate == Path("/tmp/kaggle-cli/bin/kaggle"):
                previous = environment.get("PYTHONPATH")
                environment["PYTHONPATH"] = "/tmp/kaggle-cli" + (
                    f":{previous}" if previous else ""
                )
            return candidate, environment
    raise RuntimeError(
        "Kaggle CLI was not found. Install it with: "
        "python3 -m venv .venv && .venv/bin/python -m pip install -U kaggle"
    )


def _run_kaggle(
    executable: Path,
    environment: dict[str, str],
    arguments: list[str],
    timeout: int = 120,
) -> str:
    command = [str(executable), *arguments]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Kaggle CLI did not respond within {timeout} seconds: "
            f"{' '.join(command)}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Kaggle CLI failed: {' '.join(command)}\n{detail}")
    return completed.stdout


def _parse_episode_output(output: str) -> list[dict[str, Any]]:
    """Parse JSON even when the CLI appends a human-readable hint after it."""

    try:
        value, _ = json.JSONDecoder().raw_decode(output.lstrip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("Kaggle CLI did not return a JSON episode list") from exc
    if not isinstance(value, list):
        raise RuntimeError("Kaggle CLI episode response is not a list")
    return [episode for episode in value if isinstance(episode, dict)]


def _completed_episode_ids(episodes: list[dict[str, Any]]) -> list[int]:
    ids = []
    for episode in episodes:
        if str(episode.get("state", "")).upper().endswith("COMPLETED"):
            try:
                ids.append(int(episode["id"]))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(set(ids))


def _existing_episode_ids(replay_directory: Path) -> set[int]:
    ids: set[int] = set()
    for path in replay_directory.glob("*.json"):
        match = re.search(r"\d{6,}", path.stem)
        if match:
            ids.add(int(match.group()))
    return ids


def update_replays(
    submission_id: int,
    replay_directory: Path,
    dataset_directory: Path,
    team: str,
    kaggle_path: str | None = None,
    list_only: bool = False,
    download_only: bool = False,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    executable, environment = _find_kaggle(kaggle_path)
    output = _run_kaggle(
        executable,
        environment,
        ["competitions", "episodes", str(submission_id), "--format", "json"],
    )
    remote_ids = _completed_episode_ids(_parse_episode_output(output))
    existing_ids = _existing_episode_ids(replay_directory)
    missing_ids = sorted(set(remote_ids) - existing_ids)

    result: dict[str, Any] = {
        "submission_id": submission_id,
        "remote_completed_episodes": len(remote_ids),
        "local_replays_before": len(existing_ids),
        "missing_episode_ids": missing_ids,
    }
    if list_only:
        return result

    replay_directory.mkdir(parents=True, exist_ok=True)
    for episode_id in missing_ids:
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

    if download_only:
        return result
    dataset_files_exist = all(
        (dataset_directory / filename).exists()
        for filename in ("transitions.jsonl.gz", "features.jsonl.gz")
    )
    if missing_ids or force_rebuild or not dataset_files_exist:
        manifests = build_all_datasets(replay_directory, dataset_directory, team)
        result["dataset_rebuilt"] = True
        result["manifests"] = manifests
    else:
        result["dataset_rebuilt"] = False
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-id", type=int, default=DEFAULT_SUBMISSION_ID)
    parser.add_argument("--replays", type=Path, default=Path("data/replays"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed"))
    parser.add_argument("--team", default=DEFAULT_TEAM)
    parser.add_argument("--kaggle", help="Path to the Kaggle executable")
    parser.add_argument(
        "--list-only", action="store_true", help="Show missing episode IDs only"
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Download without rebuilding datasets"
    )
    parser.add_argument(
        "--force-rebuild", action="store_true", help="Rebuild even when no replay is new"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = update_replays(
        submission_id=args.submission_id,
        replay_directory=args.replays,
        dataset_directory=args.dataset,
        team=args.team,
        kaggle_path=args.kaggle,
        list_only=args.list_only,
        download_only=args.download_only,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
