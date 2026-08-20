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
    parser.add_argument(
        "--worker-only",
        action="store_true",
        help="Build only the worker dataset needed for Worker Behavior Cloning",
    )
    parser.add_argument(
        "--reuse-transitions",
        action="store_true",
        help="Reuse a completed transitions.jsonl.gz instead of parsing replays again",
    )
    args = parser.parse_args()

    transition_path = OUTPUT_DIRECTORY / "transitions.jsonl.gz"
    manifest_path = OUTPUT_DIRECTORY / "manifest.json"
    total_stages = 2 if args.worker_only else 4
    if args.reuse_transitions:
        if not transition_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(
                "Cannot reuse transitions: transitions.jsonl.gz or manifest.json is missing"
            )
        with manifest_path.open(encoding="utf-8") as source:
            transitions = json.load(source)
        print(f"[1/{total_stages} teacher transitions] reused", flush=True)
    else:
        print(f"[1/{total_stages} teacher transitions] started", flush=True)
        transitions = build_teacher_transitions(
            REPLAY_DIRECTORY,
            OUTPUT_DIRECTORY,
            winner_only=args.winner_only,
            minimum_reward=args.minimum_reward,
            minimum_steps=args.minimum_steps,
        )
        print(f"[1/{total_stages} teacher transitions] complete", flush=True)
    if transitions["transitions"] == 0:
        raise RuntimeError(
            "No teacher trajectories passed the filters; check manifest.json "
            "or lower --minimum-steps/--minimum-reward."
        )

    if not args.worker_only:
        print("[2/4 teacher features] started", flush=True)
        features = build_feature_dataset(
            transition_path,
            OUTPUT_DIRECTORY / "features.jsonl.gz",
        )
        print("[2/4 teacher features] complete", flush=True)

    worker_stage = "2/2" if args.worker_only else "3/4"
    print(f"[{worker_stage} teacher workers] started", flush=True)
    workers = build_worker_dataset(
        transition_path,
        OUTPUT_DIRECTORY / "worker_dataset.jsonl.gz",
    )
    print(f"[{worker_stage} teacher workers] complete", flush=True)

    if not args.worker_only:
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
                "economic_rows": economics["rows"] if not args.worker_only else None,
                "features": features["feature_count"] if not args.worker_only else None,
                "output": str(OUTPUT_DIRECTORY),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
