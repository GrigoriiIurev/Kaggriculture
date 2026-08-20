"""Train and evaluate a streaming worker behavior-cloning baseline."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import SGDClassifier

from src.kaggriculture.core.action_codec import ARGUMENTS, WORKER_OPERATIONS


@dataclass(frozen=True)
class WorkerBatch:
    features: csr_matrix
    operations: np.ndarray
    arguments: np.ndarray
    is_farmer: np.ndarray


def scan_labels(path: str | Path, split: str = "train") -> dict[str, Counter[int]]:
    operation_counts: Counter[int] = Counter()
    argument_counts: Counter[int] = Counter()
    quantity_counts: Counter[int] = Counter()
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["split"] != split:
                continue
            for worker in record["workers"]:
                target = worker["target"]
                operation_counts[int(target["operation_id"])] += 1
                argument_id = int(target["argument_id"])
                if argument_id:
                    argument_counts[argument_id] += 1
                quantity_counts[int(target["quantity"])] += 1
    return {
        "operations": operation_counts,
        "arguments": argument_counts,
        "quantities": quantity_counts,
    }


def iter_batches(
    path: str | Path,
    split: str,
    feature_count: int,
    batch_size: int = 512,
) -> Iterator[WorkerBatch]:
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    operations: list[int] = []
    arguments: list[int] = []
    farmer_flags: list[bool] = []

    def make_batch() -> WorkerBatch:
        matrix = csr_matrix(
            (
                np.asarray(data, dtype=np.float32),
                np.asarray(indices, dtype=np.int32),
                np.asarray(indptr, dtype=np.int32),
            ),
            shape=(len(operations), feature_count),
        )
        return WorkerBatch(
            features=matrix,
            operations=np.asarray(operations, dtype=np.int16),
            arguments=np.asarray(arguments, dtype=np.int16),
            is_farmer=np.asarray(farmer_flags, dtype=np.bool_),
        )

    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["split"] != split:
                continue
            base_indices = record["feature_indices"]
            base_values = record["feature_values"]
            for worker in record["workers"]:
                worker_indices = worker["context_indices"]
                worker_values = worker["context_values"]
                indices.extend(base_indices)
                indices.extend(worker_indices)
                data.extend(base_values)
                data.extend(worker_values)
                indptr.append(len(indices))
                operations.append(int(worker["target"]["operation_id"]))
                arguments.append(int(worker["target"]["argument_id"]))
                farmer_flags.append(bool(worker["is_farmer"]))
                if len(operations) == batch_size:
                    yield make_batch()
                    data.clear()
                    indices.clear()
                    indptr[:] = [0]
                    operations.clear()
                    arguments.clear()
                    farmer_flags.clear()
    if operations:
        yield make_batch()


def _class_weights(counts: Counter[int], exponent: float = 0.35) -> dict[int, float]:
    if not counts:
        return {}
    largest = max(counts.values())
    weights = {
        label: (largest / count) ** exponent for label, count in counts.items()
    }
    normalizer = sum(counts[label] * weight for label, weight in weights.items())
    normalizer /= sum(counts.values())
    return {label: weight / normalizer for label, weight in weights.items()}


def _sample_weights(labels: np.ndarray, weights: dict[int, float]) -> np.ndarray:
    return np.asarray([weights[int(label)] for label in labels], dtype=np.float64)


def _with_operation_feature(
    features: csr_matrix, operations: np.ndarray
) -> csr_matrix:
    rows = np.arange(len(operations), dtype=np.int32)
    operation_features = csr_matrix(
        (
            np.ones(len(operations), dtype=np.float32),
            (rows, operations.astype(np.int32)),
        ),
        shape=(len(operations), len(WORKER_OPERATIONS)),
    )
    return hstack((features, operation_features), format="csr", dtype=np.float32)


def train_models(
    dataset_path: str | Path,
    feature_count: int,
    epochs: int = 3,
    batch_size: int = 512,
    alpha: float = 1e-5,
    seed: int = 17,
    balance_exponent: float = 0.15,
) -> tuple[SGDClassifier, SGDClassifier, dict[str, Counter[int]]]:
    labels = scan_labels(dataset_path, "train")
    operation_classes = np.asarray(sorted(labels["operations"]), dtype=np.int16)
    argument_classes = np.asarray(sorted(labels["arguments"]), dtype=np.int16)
    if len(operation_classes) < 2 or len(argument_classes) < 2:
        raise ValueError("Training split does not contain enough action classes")
    if set(labels["quantities"]) != {1}:
        raise ValueError("Worker quantity is not constant; add a quantity model")

    operation_model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        learning_rate="optimal",
        average=True,
        random_state=seed,
        shuffle=True,
    )
    argument_model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        learning_rate="optimal",
        average=True,
        random_state=seed + 1,
        shuffle=True,
    )
    operation_weights = _class_weights(labels["operations"], balance_exponent)
    argument_weights = _class_weights(labels["arguments"], balance_exponent)
    operation_initialized = False
    argument_initialized = False

    for epoch in range(epochs):
        samples = 0
        for batch in iter_batches(dataset_path, "train", feature_count, batch_size):
            operation_model.partial_fit(
                batch.features,
                batch.operations,
                classes=operation_classes if not operation_initialized else None,
                sample_weight=_sample_weights(batch.operations, operation_weights),
            )
            operation_initialized = True
            argument_mask = batch.arguments != 0
            if np.any(argument_mask):
                argument_features = _with_operation_feature(
                    batch.features[argument_mask], batch.operations[argument_mask]
                )
                argument_targets = batch.arguments[argument_mask]
                argument_model.partial_fit(
                    argument_features,
                    argument_targets,
                    classes=argument_classes if not argument_initialized else None,
                    sample_weight=_sample_weights(argument_targets, argument_weights),
                )
                argument_initialized = True
            samples += len(batch.operations)
        print(f"epoch {epoch + 1}/{epochs}: {samples} worker samples")
    return operation_model, argument_model, labels


def evaluate_models(
    dataset_path: str | Path,
    feature_count: int,
    operation_model: SGDClassifier,
    argument_model: SGDClassifier,
    batch_size: int = 1024,
) -> dict[str, object]:
    operation_support: Counter[int] = Counter()
    operation_correct: Counter[int] = Counter()
    predicted_counts: Counter[int] = Counter()
    total = operation_hits = argument_total = argument_hits = full_hits = 0
    oracle_argument_hits = 0
    farmer_total = farmer_hits = hand_total = hand_hits = 0

    for batch in iter_batches(dataset_path, "holdout", feature_count, batch_size):
        operation_predictions = operation_model.predict(batch.features)
        argument_predictions = argument_model.predict(
            _with_operation_feature(batch.features, operation_predictions)
        )
        oracle_argument_predictions = argument_model.predict(
            _with_operation_feature(batch.features, batch.operations)
        )
        operation_matches = operation_predictions == batch.operations
        argument_required = batch.arguments != 0
        argument_matches = argument_predictions == batch.arguments
        oracle_argument_matches = oracle_argument_predictions == batch.arguments
        full_matches = operation_matches & (~argument_required | argument_matches)

        total += len(batch.operations)
        operation_hits += int(np.sum(operation_matches))
        argument_total += int(np.sum(argument_required))
        argument_hits += int(np.sum(argument_matches & argument_required))
        oracle_argument_hits += int(
            np.sum(oracle_argument_matches & argument_required)
        )
        full_hits += int(np.sum(full_matches))
        farmer_total += int(np.sum(batch.is_farmer))
        farmer_hits += int(np.sum(full_matches & batch.is_farmer))
        hand_total += int(np.sum(~batch.is_farmer))
        hand_hits += int(np.sum(full_matches & ~batch.is_farmer))
        operation_support.update(int(value) for value in batch.operations)
        predicted_counts.update(int(value) for value in operation_predictions)
        operation_correct.update(
            int(value) for value in batch.operations[operation_matches]
        )

    if not total:
        raise ValueError("Holdout split is empty")
    per_operation = {}
    recalls = []
    for operation_id, support in sorted(operation_support.items()):
        recall = operation_correct[operation_id] / support
        recalls.append(recall)
        per_operation[WORKER_OPERATIONS[operation_id]] = {
            "support": support,
            "correct": operation_correct[operation_id],
            "recall": round(recall, 6),
            "predicted": predicted_counts[operation_id],
        }
    majority = max(operation_support.values()) / total
    return {
        "holdout_worker_samples": total,
        "operation_accuracy": round(operation_hits / total, 6),
        "operation_balanced_accuracy": round(sum(recalls) / len(recalls), 6),
        "argument_samples": argument_total,
        "argument_accuracy": round(argument_hits / max(1, argument_total), 6),
        "argument_accuracy_given_true_operation": round(
            oracle_argument_hits / max(1, argument_total), 6
        ),
        "full_command_accuracy": round(full_hits / total, 6),
        "farmer_full_accuracy": round(farmer_hits / max(1, farmer_total), 6),
        "hand_full_accuracy": round(hand_hits / max(1, hand_total), 6),
        "majority_operation_baseline": round(majority, 6),
        "per_operation": per_operation,
    }


def export_model(
    path: str | Path,
    feature_count: int,
    operation_model: SGDClassifier,
    argument_model: SGDClassifier,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.asarray([1], dtype=np.int16),
        feature_count=np.asarray([feature_count], dtype=np.int32),
        operation_classes=operation_model.classes_.astype(np.int16),
        operation_coef=operation_model.coef_.astype(np.float32),
        operation_intercept=operation_model.intercept_.astype(np.float32),
        argument_classes=argument_model.classes_.astype(np.int16),
        argument_coef=argument_model.coef_.astype(np.float32),
        argument_intercept=argument_model.intercept_.astype(np.float32),
        argument_feature_count=np.asarray(
            [feature_count + len(WORKER_OPERATIONS)], dtype=np.int32
        ),
    )


def _json_counts(counts: Counter[int], names: tuple[str, ...]) -> dict[str, int]:
    return {names[index]: count for index, count in sorted(counts.items())}


def train_behavior_cloning(
    dataset_path: str | Path,
    worker_manifest_path: str | Path,
    transitions_path: str | Path,
    model_path: str | Path,
    report_path: str | Path,
    policy_report_path: str | Path,
    epochs: int = 3,
    batch_size: int = 512,
    alpha: float = 1e-5,
    seed: int = 17,
    balance_exponent: float = 0.15,
) -> dict[str, object]:
    with Path(worker_manifest_path).open(encoding="utf-8") as source:
        worker_manifest = json.load(source)
    feature_count = int(worker_manifest["feature_count"])
    operation_model, argument_model, labels = train_models(
        dataset_path,
        feature_count,
        epochs=epochs,
        batch_size=batch_size,
        alpha=alpha,
        seed=seed,
        balance_exponent=balance_exponent,
    )
    metrics = evaluate_models(
        dataset_path,
        feature_count,
        operation_model,
        argument_model,
        batch_size=max(batch_size, 1024),
    )
    export_model(model_path, feature_count, operation_model, argument_model)
    from .evaluate_behavior_policy import evaluate_policy

    policy_metrics = evaluate_policy(transitions_path, model_path)
    policy_report_path = Path(policy_report_path)
    policy_report_path.parent.mkdir(parents=True, exist_ok=True)
    with policy_report_path.open("w", encoding="utf-8") as output:
        json.dump(policy_metrics, output, ensure_ascii=True, indent=2)
        output.write("\n")
    report = {
        "model": "streaming_sgd_logistic_behavior_cloning",
        "dataset": str(Path(dataset_path).resolve()),
        "model_file": str(Path(model_path)),
        "feature_count": feature_count,
        "epochs": epochs,
        "batch_size": batch_size,
        "alpha": alpha,
        "seed": seed,
        "balance_exponent": balance_exponent,
        "train_operation_counts": _json_counts(
            labels["operations"], WORKER_OPERATIONS
        ),
        "train_argument_counts": _json_counts(labels["arguments"], ARGUMENTS),
        "metrics": metrics,
        "masked_policy_metrics": policy_metrics,
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=True, indent=2)
        output.write("\n")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="data/processed/worker_dataset.jsonl.gz"
    )
    parser.add_argument(
        "--manifest", default="data/processed/worker_manifest.json"
    )
    parser.add_argument(
        "--transitions", default="data/processed/transitions.jsonl.gz"
    )
    parser.add_argument("--model", default="experiments/behavior_cloning/artifacts/worker_bc.npz")
    parser.add_argument("--report", default="experiments/behavior_cloning/artifacts/worker_bc_report.json")
    parser.add_argument(
        "--policy-report", default="experiments/behavior_cloning/artifacts/worker_bc_policy_report.json"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--balance-exponent", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = train_behavior_cloning(
        args.dataset,
        args.manifest,
        args.transitions,
        args.model,
        args.report,
        args.policy_report,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        seed=args.seed,
        balance_exponent=args.balance_exponent,
    )
    print(
        json.dumps(
            {
                "offline_metrics": report["metrics"],
                "masked_policy_metrics": report["masked_policy_metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
