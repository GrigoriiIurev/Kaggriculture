"""Build all learning files from manually downloaded teacher replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.kaggriculture.data.economic_dataset import build_economic_dataset
from src.kaggriculture.data.feature_extractor import build_feature_dataset
from src.kaggriculture.data.teacher_dataset import build_teacher_transitions
from src.kaggriculture.data.worker_dataset import build_worker_dataset


REPLAY_DIRECTORY = Path("data/teacher_replays")
OUTPUT_DIRECTORY = Path("data/teacher_processed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner-only", action="store_true")
    parser.add_argument("--minimum-reward", type=float, default=0.0)
    parser.add_argument("--minimum-steps", type=int, default=700)
    args = parser.parse_args()

    print("[1/4 teacher transitions] started", flush=True)
    transitions = build_teacher_transitions(
        REPLAY_DIRECTORY,
        OUTPUT_DIRECTORY,
        winner_only=args.winner_only,
        minimum_reward=args.minimum_reward,
        minimum_steps=args.minimum_steps,
    )
    print("[1/4 teacher transitions] complete", flush=True)
    if transitions["transitions"] == 0:
        raise RuntimeError(
            "No teacher trajectories passed the filters; check manifest.json "
            "or lower --minimum-steps/--minimum-reward."
        )

    print("[2/4 teacher features] started", flush=True)
    features = build_feature_dataset(
        OUTPUT_DIRECTORY / "transitions.jsonl.gz",
        OUTPUT_DIRECTORY / "features.jsonl.gz",
    )
    print("[2/4 teacher features] complete", flush=True)

    print("[3/4 teacher workers] started", flush=True)
    workers = build_worker_dataset(
        OUTPUT_DIRECTORY / "transitions.jsonl.gz",
        OUTPUT_DIRECTORY / "worker_dataset.jsonl.gz",
    )
    print("[3/4 teacher workers] complete", flush=True)

    print("[4/4 teacher economics] started", flush=True)
    economics = build_economic_dataset(
        OUTPUT_DIRECTORY / "features.jsonl.gz",
        OUTPUT_DIRECTORY / "economic_dataset.jsonl.gz",
    )
    print("[4/4 teacher economics] complete", flush=True)

    print(
        json.dumps(
            {
                "episodes": transitions["unique_episodes_seen"],
                "trajectories": transitions["selected_trajectories"],
                "transitions": transitions["transitions"],
                "worker_samples": workers["worker_samples"],
                "economic_rows": economics["rows"],
                "features": features["feature_count"],
                "output": str(OUTPUT_DIRECTORY),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
