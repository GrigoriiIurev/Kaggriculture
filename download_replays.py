"""Download only the missing replays for the current Kaggle submission."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.update_replays import DEFAULT_SUBMISSION_ID, DEFAULT_TEAM, update_replays


def main() -> None:
    result = update_replays(
        submission_id=DEFAULT_SUBMISSION_ID,
        replay_directory=Path("data/replays"),
        dataset_directory=Path("data/processed"),
        team=DEFAULT_TEAM,
        download_only=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
