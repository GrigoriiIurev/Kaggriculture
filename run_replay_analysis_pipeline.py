"""Synchronize submission replays and build the Replay Warehouse on Drive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.sync_submission_replays import (
    _parse_json_list,
    select_latest_completed_submission,
    sync_submission_replays,
)
from scripts.update_replays import _find_kaggle, _run_kaggle
from src.kaggriculture.analysis.replay_warehouse import build_replay_warehouse
from src.kaggriculture.analysis.loss_replay_analyzer import (
    build_loss_replay_analysis,
)


def resolve_submission(
    competition: str,
    submission_id: int | None,
    kaggle_path: str | None = None,
) -> dict[str, object]:
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
        return select_latest_completed_submission(submissions)
    return next(
        (item for item in submissions if int(item.get("ref", -1)) == submission_id),
        {"ref": submission_id},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--submission-id", type=int)
    parser.add_argument("--competition", default="kaggriculture")
    parser.add_argument("--team")
    parser.add_argument("--max-replays", type=int, default=0)
    parser.add_argument("--force-analysis", action="store_true")
    parser.add_argument("--kaggle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[1/4] Resolving the Kaggle submission", flush=True)
    submission = resolve_submission(
        args.competition, args.submission_id, args.kaggle
    )
    submission_id = int(submission["ref"])
    warehouse = args.drive_root / "replay_warehouse" / f"submission_{submission_id}"
    replays = warehouse / "replays"
    index = warehouse / "episode_index.json"
    analysis = warehouse / "analysis"

    print("[2/4] Synchronizing completed replays", flush=True)
    sync_result = sync_submission_replays(
        replay_directory=replays,
        index_path=index,
        submission_id=submission_id,
        competition=args.competition,
        kaggle_path=args.kaggle,
        max_replays=args.max_replays,
    )

    manifest_path = analysis / "manifest.json"
    previous_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    needs_analysis = (
        args.force_analysis
        or sync_result["downloaded_now"] > 0
        or previous_manifest is None
        or previous_manifest.get("unique_episodes") != sync_result["local_selected"]
        or (args.team and previous_manifest.get("team_name") != args.team)
    )
    print("[3/4] Building compact analytical datasets", flush=True)
    if needs_analysis:
        manifest = build_replay_warehouse(
            replay_directory=replays,
            output_directory=analysis,
            team_name=args.team,
            episode_index_path=index,
        )
    else:
        manifest = previous_manifest
        print("[warehouse] No new replay; reusing the existing analysis", flush=True)

    print("[4/4] Diagnosing losses and producing experiments", flush=True)
    loss_report = build_loss_replay_analysis(
        analysis / manifest["files"]["daily_macro"],
        analysis,
        str(manifest["team_name"]),
    )

    receipt = {
        "submission": submission,
        "warehouse": str(warehouse),
        "replays": str(replays),
        "analysis": str(analysis),
        "sync": sync_result,
        "summary": manifest["summary"],
        "loss_summary": loss_report["summary"],
        "files": manifest["files"],
        "loss_files": loss_report["files"],
    }
    latest = args.drive_root / "replay_warehouse" / "latest_analysis.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
