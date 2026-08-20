"""Build every dataset from the replay directory with one command."""

from __future__ import annotations

import json
from pathlib import Path

from src.kaggriculture.data.pipeline import build_all_datasets


TEAM_NAME = "Grigorii IU"
REPLAY_DIRECTORY = Path("data/replays")
DATASET_DIRECTORY = Path("data/processed")


def main() -> None:
    replay_files = list(REPLAY_DIRECTORY.glob("*.json"))
    print(f"Found {len(replay_files)} replay files. Building all datasets...")
    manifests = build_all_datasets(REPLAY_DIRECTORY, DATASET_DIRECTORY, TEAM_NAME)
    result = {
        "episodes": manifests["replays"]["unique_episodes_used"],
        "transitions": manifests["replays"]["transitions"],
        "worker_samples": manifests["workers"]["worker_samples"],
        "economic_rows": manifests["economics"]["rows"],
        "value_rows": manifests["value"]["records"],
        "output": str(DATASET_DIRECTORY),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
