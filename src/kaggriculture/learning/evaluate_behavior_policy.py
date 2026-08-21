"""Evaluate the exported, legally masked worker policy on holdout episodes."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from src.kaggriculture.core.action_codec import ActionEncoder
from src.kaggriculture.learning.behavior_model import BehaviorCloningPolicy
from src.kaggriculture.core.legal_actions import (
    ARGUMENT_OPERATIONS,
    legal_worker_arguments,
    legal_worker_operations,
)
from src.kaggriculture.core.state_parser import parse_observation
from src.kaggriculture.data.worker_dataset import episode_split


def evaluate_policy(
    transitions_path: str | Path,
    model_path: str | Path,
) -> dict[str, object]:
    policy = BehaviorCloningPolicy(model_path)
    encoder = ActionEncoder()
    groups = worker_samples = operation_hits = full_hits = excluded_targets = 0
    transitions_seen = 0
    next_progress = 10_000
    episodes: set[int] = set()

    with gzip.open(transitions_path, "rt", encoding="utf-8") as source:
        for line in source:
            transition = json.loads(line)
            transitions_seen += 1
            if transitions_seen >= next_progress:
                print(
                    "[policy evaluation] "
                    f"{transitions_seen:,} transitions scanned, "
                    f"{worker_samples:,} holdout worker samples checked",
                    flush=True,
                )
                next_progress += 10_000
            episode_id = int(transition["episode_id"])
            if episode_split(episode_id) != "holdout":
                continue
            groups += 1
            episodes.add(episode_id)
            state = parse_observation(transition["observation"])
            predictions = policy.predict_commands(transition["observation"])
            targets = (
                transition["action"]["farmer"],
                *transition["action"]["hands"],
            )
            for worker_index, (target, prediction) in enumerate(
                zip(targets, predictions)
            ):
                encoded_target = encoder.encode_worker(target)
                encoded_prediction = encoder.encode_worker(prediction)
                worker_samples += 1
                operation_hits += (
                    encoded_target.operation_id == encoded_prediction.operation_id
                )
                full_hits += encoded_target == encoded_prediction
                legal_operations = legal_worker_operations(state, worker_index)
                if target[0] not in legal_operations:
                    excluded_targets += 1
                elif target[0] in ARGUMENT_OPERATIONS:
                    legal_arguments = legal_worker_arguments(
                        state, worker_index, target[0]
                    )
                    excluded_targets += target[1] not in legal_arguments

    if not worker_samples:
        raise ValueError("Holdout split is empty")
    return {
        "holdout_episodes": len(episodes),
        "holdout_transition_groups": groups,
        "holdout_worker_samples": worker_samples,
        "masked_operation_accuracy": round(operation_hits / worker_samples, 6),
        "masked_full_command_accuracy": round(full_hits / worker_samples, 6),
        "ground_truth_commands_excluded_by_mask": excluded_targets,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transitions", default="data/processed/transitions.jsonl.gz"
    )
    parser.add_argument("--model", default="experiments/behavior_cloning/artifacts/worker_bc.npz")
    parser.add_argument("--output", default="experiments/behavior_cloning/artifacts/worker_bc_policy_report.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = evaluate_policy(args.transitions, args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=True, indent=2)
        output.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
