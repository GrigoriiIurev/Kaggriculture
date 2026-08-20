"""Train reward-weighted and class-balanced multi-slot economic BC."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy import sparse
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

from src.kaggriculture.core.action_codec import (
    ARGUMENTS,
    MARKET_NO_ARGUMENT_OPERATIONS,
    MARKET_OPERATIONS,
)


DEFAULT_DATASET = "data/processed/economic_dataset.jsonl.gz"
DEFAULT_FEATURE_SCHEMA = "data/processed/feature_schema.json"
DEFAULT_MODEL = "experiments/behavior_cloning/artifacts/economic_bc.npz"
DEFAULT_REPORT = "experiments/behavior_cloning/artifacts/economic_bc_report.json"

MAX_ORDERS = 10

DEFAULT_BATCH_SIZE = 4096
DEFAULT_EPOCHS = 8
DEFAULT_SEED = 17

NO_ORDER_ID = MARKET_OPERATIONS.index("NO_ORDER")

QUANTITY_OPERATIONS = {
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
}

# Prevent one rare class from receiving absurdly large influence.
MAX_CLASS_WEIGHT = 4.0
MIN_CLASS_WEIGHT = 0.75

# Final training sample weight is clipped here.
MIN_SAMPLE_WEIGHT = 0.15
MAX_SAMPLE_WEIGHT = 6.0


def read_feature_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    feature_count = int(schema["feature_count"])

    if feature_count <= 0:
        raise ValueError("feature_count must be positive")

    return feature_count


def iter_rows(
    path: Path,
    split: str,
) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            if row.get("split") == split:
                yield row


def target_for_slot(
    row: dict[str, Any],
    slot: int,
) -> dict[str, Any]:
    targets = row.get("targets", [])

    if slot < len(targets):
        return targets[slot]

    return {
        "operation_id": NO_ORDER_ID,
        "argument_id": 0,
        "quantity": 0,
    }


def needs_argument(operation_id: int) -> bool:
    operation = MARKET_OPERATIONS[operation_id]

    return (
        operation != "NO_ORDER"
        and operation not in MARKET_NO_ARGUMENT_OPERATIONS
    )


def needs_quantity(operation_id: int) -> bool:
    return MARKET_OPERATIONS[operation_id] in QUANTITY_OPERATIONS


# ---------------------------------------------------------------------
# Episode-quality weighting
# ---------------------------------------------------------------------


def collect_episode_quality(
    dataset: Path,
) -> dict[int, float]:
    """
    Produce one quality weight per training episode.

    Episodes are ranked by margin instead of relying on its raw scale.
    This makes weighting robust to large outliers.

    Best trajectories receive substantially more training weight.
    """

    episodes: dict[int, dict[str, Any]] = {}

    for row in iter_rows(dataset, "train"):
        episode_id = int(row["episode_id"])

        if episode_id in episodes:
            continue

        try:
            margin = float(row.get("margin", 0.0) or 0.0)
        except (TypeError, ValueError):
            margin = 0.0

        if not math.isfinite(margin):
            margin = 0.0

        episodes[episode_id] = {
            "margin": margin,
            "outcome": str(
                row.get("outcome", "")
            ).upper(),
        }

    if not episodes:
        raise RuntimeError("No train episodes found")

    ordered = sorted(
        episodes.items(),
        key=lambda item: item[1]["margin"],
    )

    n = len(ordered)

    weights: dict[int, float] = {}

    for rank, (episode_id, info) in enumerate(ordered):
        if n == 1:
            percentile = 0.5
        else:
            percentile = rank / (n - 1)

        # Strong preference for top trajectories.
        #
        # bottom  -> ~0.40
        # median  -> ~0.93
        # top     -> 2.50
        rank_weight = (
            0.40
            + 2.10 * percentile**2
        )

        outcome = info["outcome"]

        if outcome in {"WIN", "WON", "VICTORY"}:
            outcome_multiplier = 1.35

        elif outcome in {
            "LOSS",
            "LOSE",
            "LOST",
            "DEFEAT",
        }:
            outcome_multiplier = 0.70

        else:
            outcome_multiplier = 1.0

        weight = (
            rank_weight
            * outcome_multiplier
        )

        weights[episode_id] = float(
            np.clip(
                weight,
                0.25,
                3.5,
            )
        )

    return weights


# ---------------------------------------------------------------------
# Class balancing
# ---------------------------------------------------------------------


def collect_class_statistics(
    dataset: Path,
):
    operation_counts = [
        Counter()
        for _ in range(MAX_ORDERS)
    ]

    argument_counts = [
        Counter()
        for _ in range(MAX_ORDERS)
    ]

    for row in iter_rows(dataset, "train"):
        for slot in range(MAX_ORDERS):
            target = target_for_slot(
                row,
                slot,
            )

            operation_id = int(
                target["operation_id"]
            )

            operation_counts[slot][
                operation_id
            ] += 1

            if needs_argument(operation_id):
                argument_id = int(
                    target["argument_id"]
                )

                argument_counts[slot][
                    argument_id
                ] += 1

    return (
        operation_counts,
        argument_counts,
    )


def make_balanced_weights(
    counts: Counter,
) -> dict[int, float]:
    """
    Inverse-square-root class weighting.

    Much gentler than full inverse frequency:
        weight ~ sqrt(max_count / class_count)
    """

    if not counts:
        return {}

    maximum = max(counts.values())

    result = {}

    for class_id, count in counts.items():
        if count <= 0:
            continue

        weight = math.sqrt(
            maximum / count
        )

        result[class_id] = float(
            np.clip(
                weight,
                MIN_CLASS_WEIGHT,
                MAX_CLASS_WEIGHT,
            )
        )

    return result


def build_class_weights(
    operation_counts,
    argument_counts,
):
    operation_weights = [
        make_balanced_weights(counts)
        for counts in operation_counts
    ]

    argument_weights = [
        make_balanced_weights(counts)
        for counts in argument_counts
    ]

    return (
        operation_weights,
        argument_weights,
    )


# ---------------------------------------------------------------------
# Sparse matrices
# ---------------------------------------------------------------------


def rows_to_matrix(
    rows: list[dict[str, Any]],
    feature_count: int,
) -> sparse.csr_matrix:
    data = []
    indices = []
    indptr = [0]

    for row in rows:
        feature_indices = row[
            "feature_indices"
        ]

        feature_values = row[
            "feature_values"
        ]

        if len(feature_indices) != len(
            feature_values
        ):
            raise ValueError(
                "feature_indices and "
                "feature_values have different lengths"
            )

        for index, value in zip(
            feature_indices,
            feature_values,
        ):
            index = int(index)

            if not 0 <= index < feature_count:
                raise ValueError(
                    f"Feature index {index} outside "
                    f"[0, {feature_count})"
                )

            indices.append(index)
            data.append(float(value))

        indptr.append(len(indices))

    return sparse.csr_matrix(
        (
            np.asarray(
                data,
                dtype=np.float32,
            ),
            np.asarray(
                indices,
                dtype=np.int32,
            ),
            np.asarray(
                indptr,
                dtype=np.int32,
            ),
        ),
        shape=(
            len(rows),
            feature_count,
        ),
        dtype=np.float32,
    )


def iter_batches(
    path: Path,
    split: str,
    batch_size: int,
):
    batch = []

    for row in iter_rows(path, split):
        batch.append(row)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------


def new_classifier(
    seed: int,
) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=2e-5,
        learning_rate="optimal",
        average=True,
        random_state=seed,
    )


def new_regressor(
    seed: int,
) -> SGDRegressor:
    return SGDRegressor(
        loss="huber",
        penalty="l2",
        alpha=2e-5,
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        average=True,
        random_state=seed,
    )


class SlotModels:
    def __init__(
        self,
        slot: int,
        seed: int,
    ):
        offset = slot * 100

        self.operation = new_classifier(
            seed + offset
        )

        self.argument = new_classifier(
            seed + offset + 1
        )

        self.quantity = new_regressor(
            seed + offset + 2
        )

        self.operation_initialized = False
        self.argument_initialized = False
        self.quantity_initialized = False


def combined_weight(
    episode_weight: float,
    class_weight: float,
) -> float:
    return float(
        np.clip(
            episode_weight * class_weight,
            MIN_SAMPLE_WEIGHT,
            MAX_SAMPLE_WEIGHT,
        )
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def train(
    dataset: Path,
    feature_count: int,
    epochs: int,
    batch_size: int,
    seed: int,
    episode_weights: dict[int, float],
    operation_weights,
    argument_weights,
):
    slots = [
        SlotModels(slot, seed)
        for slot in range(MAX_ORDERS)
    ]

    operation_classes = np.arange(
        len(MARKET_OPERATIONS),
        dtype=np.int64,
    )

    argument_classes = np.arange(
        len(ARGUMENTS),
        dtype=np.int64,
    )

    epoch_reports = []

    for epoch in range(
        1,
        epochs + 1,
    ):
        row_count = 0

        active_counts = np.zeros(
            MAX_ORDERS,
            dtype=np.int64,
        )

        weight_sum = 0.0
        weight_count = 0

        for rows in iter_batches(
            dataset,
            "train",
            batch_size,
        ):
            X = rows_to_matrix(
                rows,
                feature_count,
            )

            episode_batch_weights = np.asarray(
                [
                    episode_weights[
                        int(row["episode_id"])
                    ]
                    for row in rows
                ],
                dtype=np.float64,
            )

            for (
                slot_index,
                models,
            ) in enumerate(slots):

                targets = [
                    target_for_slot(
                        row,
                        slot_index,
                    )
                    for row in rows
                ]

                y_operation = np.asarray(
                    [
                        int(
                            target[
                                "operation_id"
                            ]
                        )
                        for target in targets
                    ],
                    dtype=np.int64,
                )

                op_sample_weights = np.asarray(
                    [
                        combined_weight(
                            episode_weight,
                            operation_weights[
                                slot_index
                            ].get(
                                int(operation_id),
                                1.0,
                            ),
                        )
                        for (
                            episode_weight,
                            operation_id,
                        ) in zip(
                            episode_batch_weights,
                            y_operation,
                        )
                    ],
                    dtype=np.float64,
                )

                if not models.operation_initialized:
                    models.operation.partial_fit(
                        X,
                        y_operation,
                        classes=operation_classes,
                        sample_weight=op_sample_weights,
                    )

                    models.operation_initialized = True

                else:
                    models.operation.partial_fit(
                        X,
                        y_operation,
                        sample_weight=op_sample_weights,
                    )

                active_counts[
                    slot_index
                ] += int(
                    np.sum(
                        y_operation
                        != NO_ORDER_ID
                    )
                )

                weight_sum += float(
                    op_sample_weights.sum()
                )

                weight_count += len(
                    op_sample_weights
                )

                # -------------------------------------------------
                # Argument model
                # -------------------------------------------------

                argument_mask = np.asarray(
                    [
                        needs_argument(op)
                        for op in y_operation
                    ],
                    dtype=bool,
                )

                if np.any(argument_mask):
                    y_argument = np.asarray(
                        [
                            int(
                                target[
                                    "argument_id"
                                ]
                            )
                            for target in targets
                        ],
                        dtype=np.int64,
                    )

                    argument_ids = (
                        y_argument[
                            argument_mask
                        ]
                    )

                    argument_episode_weights = (
                        episode_batch_weights[
                            argument_mask
                        ]
                    )

                    argument_sample_weights = (
                        np.asarray(
                            [
                                combined_weight(
                                    episode_weight,
                                    argument_weights[
                                        slot_index
                                    ].get(
                                        int(argument_id),
                                        1.0,
                                    ),
                                )
                                for (
                                    episode_weight,
                                    argument_id,
                                ) in zip(
                                    argument_episode_weights,
                                    argument_ids,
                                )
                            ],
                            dtype=np.float64,
                        )
                    )

                    if not models.argument_initialized:
                        models.argument.partial_fit(
                            X[argument_mask],
                            argument_ids,
                            classes=argument_classes,
                            sample_weight=(
                                argument_sample_weights
                            ),
                        )

                        models.argument_initialized = True

                    else:
                        models.argument.partial_fit(
                            X[argument_mask],
                            argument_ids,
                            sample_weight=(
                                argument_sample_weights
                            ),
                        )

                # -------------------------------------------------
                # Quantity model
                # -------------------------------------------------

                quantity_mask = np.asarray(
                    [
                        needs_quantity(op)
                        for op in y_operation
                    ],
                    dtype=bool,
                )

                if np.any(quantity_mask):
                    y_quantity = np.asarray(
                        [
                            float(
                                target[
                                    "quantity"
                                ]
                            )
                            for target in targets
                        ],
                        dtype=np.float32,
                    )

                    quantity_targets = np.log1p(
                        y_quantity[
                            quantity_mask
                        ]
                    )

                    # Reward + operation balance also applies to quantity.
                    quantity_weights = (
                        op_sample_weights[
                            quantity_mask
                        ]
                    )

                    models.quantity.partial_fit(
                        X[quantity_mask],
                        quantity_targets,
                        sample_weight=(
                            quantity_weights
                        ),
                    )

                    models.quantity_initialized = True

            row_count += len(rows)

        mean_weight = (
            weight_sum / weight_count
            if weight_count
            else 0.0
        )

        epoch_report = {
            "epoch": epoch,
            "turns": row_count,
            "active_orders_per_slot": (
                active_counts.tolist()
            ),
            "mean_operation_weight": (
                mean_weight
            ),
        }

        epoch_reports.append(
            epoch_report
        )

        print(
            f"epoch {epoch}/{epochs}: "
            f"turns={row_count}, "
            f"mean_weight={mean_weight:.3f}, "
            "orders="
            + ",".join(
                str(int(x))
                for x in active_counts
            )
        )

    return slots, epoch_reports


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def evaluate(
    dataset: Path,
    feature_count: int,
    slots: list[SlotModels],
    batch_size: int,
) -> dict[str, Any]:

    slot_reports = []

    # Evaluate individual slots.
    for (
        slot_index,
        models,
    ) in enumerate(slots):

        operation_true = []
        operation_pred = []

        active_true = []
        active_pred = []

        argument_true = []
        argument_pred = []

        quantity_true = []
        quantity_pred = []

        for rows in iter_batches(
            dataset,
            "holdout",
            batch_size,
        ):
            X = rows_to_matrix(
                rows,
                feature_count,
            )

            targets = [
                target_for_slot(
                    row,
                    slot_index,
                )
                for row in rows
            ]

            y_operation = np.asarray(
                [
                    int(
                        target[
                            "operation_id"
                        ]
                    )
                    for target in targets
                ],
                dtype=np.int64,
            )

            pred_operation = (
                models.operation.predict(X)
            )

            operation_true.extend(
                y_operation.tolist()
            )

            operation_pred.extend(
                pred_operation.tolist()
            )

            active_mask = (
                y_operation
                != NO_ORDER_ID
            )

            if np.any(active_mask):
                active_true.extend(
                    y_operation[
                        active_mask
                    ].tolist()
                )

                active_pred.extend(
                    pred_operation[
                        active_mask
                    ].tolist()
                )

            argument_mask = np.asarray(
                [
                    needs_argument(op)
                    for op in y_operation
                ],
                dtype=bool,
            )

            if (
                models.argument_initialized
                and np.any(argument_mask)
            ):
                y_argument = np.asarray(
                    [
                        int(
                            target[
                                "argument_id"
                            ]
                        )
                        for target in targets
                    ],
                    dtype=np.int64,
                )

                prediction = (
                    models.argument.predict(
                        X[argument_mask]
                    )
                )

                argument_true.extend(
                    y_argument[
                        argument_mask
                    ].tolist()
                )

                argument_pred.extend(
                    prediction.tolist()
                )

            quantity_mask = np.asarray(
                [
                    needs_quantity(op)
                    for op in y_operation
                ],
                dtype=bool,
            )

            if (
                models.quantity_initialized
                and np.any(quantity_mask)
            ):
                y_quantity = np.asarray(
                    [
                        float(
                            target[
                                "quantity"
                            ]
                        )
                        for target in targets
                    ],
                    dtype=np.float32,
                )

                pred_log = (
                    models.quantity.predict(
                        X[quantity_mask]
                    )
                )

                pred_quantity = np.clip(
                    np.rint(
                        np.expm1(
                            pred_log
                        )
                    ),
                    1,
                    100,
                )

                quantity_true.extend(
                    y_quantity[
                        quantity_mask
                    ].tolist()
                )

                quantity_pred.extend(
                    pred_quantity.tolist()
                )

        if operation_true:
            predicted_active_rate = float(
                np.mean(
                    np.asarray(
                        operation_pred
                    )
                    != NO_ORDER_ID
                )
            )

            true_active_rate = float(
                np.mean(
                    np.asarray(
                        operation_true
                    )
                    != NO_ORDER_ID
                )
            )

        else:
            predicted_active_rate = None
            true_active_rate = None

        slot_reports.append(
            {
                "slot": slot_index,

                "operation_accuracy": (
                    float(
                        accuracy_score(
                            operation_true,
                            operation_pred,
                        )
                    )
                    if operation_true
                    else None
                ),

                # Much more useful than overall accuracy,
                # because NO_ORDER dominates late slots.
                "active_operation_accuracy": (
                    float(
                        accuracy_score(
                            active_true,
                            active_pred,
                        )
                    )
                    if active_true
                    else None
                ),

                "argument_accuracy": (
                    float(
                        accuracy_score(
                            argument_true,
                            argument_pred,
                        )
                    )
                    if argument_true
                    else None
                ),

                "quantity_mae": (
                    float(
                        mean_absolute_error(
                            quantity_true,
                            quantity_pred,
                        )
                    )
                    if quantity_true
                    else None
                ),

                "real_orders": len(
                    active_true
                ),

                "true_active_rate": (
                    true_active_rate
                ),

                "predicted_active_rate": (
                    predicted_active_rate
                ),
            }
        )

    # -------------------------------------------------------------
    # Exact operation-sequence accuracy
    # -------------------------------------------------------------

    exact_correct = 0
    turn_count = 0

    for rows in iter_batches(
        dataset,
        "holdout",
        batch_size,
    ):
        X = rows_to_matrix(
            rows,
            feature_count,
        )

        predictions = [
            models.operation.predict(X)
            for models in slots
        ]

        for row_index, row in enumerate(rows):
            true_sequence = [
                int(
                    target_for_slot(
                        row,
                        slot,
                    )["operation_id"]
                )
                for slot in range(
                    MAX_ORDERS
                )
            ]

            pred_sequence = [
                int(
                    predictions[
                        slot
                    ][row_index]
                )
                for slot in range(
                    MAX_ORDERS
                )
            ]

            if (
                pred_sequence
                == true_sequence
            ):
                exact_correct += 1

            turn_count += 1

    return {
        "turns": turn_count,

        "exact_operation_sequence_accuracy": (
            exact_correct / turn_count
            if turn_count
            else None
        ),

        "slots": slot_reports,
    }


# ---------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------


def save_model(
    path: Path,
    feature_count: int,
    slots: list[SlotModels],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arrays: dict[str, Any] = {
        "version": np.asarray(
            [3],
            dtype=np.int32,
        ),

        "feature_count": np.asarray(
            [feature_count],
            dtype=np.int32,
        ),

        "max_orders": np.asarray(
            [MAX_ORDERS],
            dtype=np.int32,
        ),

        "market_operations": np.asarray(
            MARKET_OPERATIONS,
            dtype=np.str_,
        ),

        "arguments": np.asarray(
            ARGUMENTS,
            dtype=np.str_,
        ),

        "quantity_transform": np.asarray(
            ["log1p"],
            dtype=np.str_,
        ),
    }

    for (
        slot_index,
        models,
    ) in enumerate(slots):

        prefix = (
            f"slot_{slot_index}"
        )

        arrays[
            f"{prefix}_operation_coef"
        ] = np.asarray(
            models.operation.coef_,
            dtype=np.float32,
        )

        arrays[
            f"{prefix}_operation_intercept"
        ] = np.asarray(
            models.operation.intercept_,
            dtype=np.float32,
        )

        arrays[
            f"{prefix}_argument_initialized"
        ] = np.asarray(
            [
                models.argument_initialized
            ],
            dtype=np.bool_,
        )

        if models.argument_initialized:
            arrays[
                f"{prefix}_argument_coef"
            ] = np.asarray(
                models.argument.coef_,
                dtype=np.float32,
            )

            arrays[
                f"{prefix}_argument_intercept"
            ] = np.asarray(
                models.argument.intercept_,
                dtype=np.float32,
            )

        arrays[
            f"{prefix}_quantity_initialized"
        ] = np.asarray(
            [
                models.quantity_initialized
            ],
            dtype=np.bool_,
        )

        if models.quantity_initialized:
            arrays[
                f"{prefix}_quantity_coef"
            ] = np.asarray(
                models.quantity.coef_,
                dtype=np.float32,
            )

            arrays[
                f"{prefix}_quantity_intercept"
            ] = np.asarray(
                models.quantity.intercept_,
                dtype=np.float32,
            )

    np.savez_compressed(
        path,
        **arrays,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--feature-schema",
        default=DEFAULT_FEATURE_SCHEMA,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()

    dataset = Path(
        args.dataset
    )

    feature_schema = Path(
        args.feature_schema
    )

    model_path = Path(
        args.model
    )

    report_path = Path(
        args.report
    )

    feature_count = (
        read_feature_count(
            feature_schema
        )
    )

    print(
        "Collecting training statistics..."
    )

    episode_weights = (
        collect_episode_quality(
            dataset
        )
    )

    (
        operation_counts,
        argument_counts,
    ) = collect_class_statistics(
        dataset
    )

    (
        operation_weights,
        argument_weights,
    ) = build_class_weights(
        operation_counts,
        argument_counts,
    )

    print(
        f"Train episodes: "
        f"{len(episode_weights)}"
    )

    episode_weight_values = list(
        episode_weights.values()
    )

    print(
        "Episode weights: "
        f"min={min(episode_weight_values):.3f}, "
        f"mean={np.mean(episode_weight_values):.3f}, "
        f"max={max(episode_weight_values):.3f}"
    )

    print()
    print(
        "Training reward-weighted + "
        "class-balanced economic model"
    )

    print(
        f"dataset: {dataset}"
    )

    print(
        f"features: {feature_count}"
    )

    print(
        f"slots: {MAX_ORDERS}"
    )

    print(
        f"epochs: {args.epochs}"
    )

    print(
        f"batch size: {args.batch_size}"
    )

    (
        slots,
        epoch_reports,
    ) = train(
        dataset=dataset,
        feature_count=feature_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        episode_weights=episode_weights,
        operation_weights=operation_weights,
        argument_weights=argument_weights,
    )

    print()
    print(
        "Evaluating on holdout..."
    )

    holdout = evaluate(
        dataset,
        feature_count,
        slots,
        args.batch_size,
    )

    save_model(
        model_path,
        feature_count,
        slots,
    )

    report = {
        "version": 3,

        "architecture": (
            "reward_weighted_"
            "class_balanced_"
            "multi_slot_bc"
        ),

        "max_orders": MAX_ORDERS,

        "feature_count": (
            feature_count
        ),

        "epochs": args.epochs,

        "batch_size": (
            args.batch_size
        ),

        "seed": args.seed,

        "episode_weighting": {
            "method": (
                "margin_rank_plus_outcome"
            ),

            "episode_count": (
                len(
                    episode_weights
                )
            ),

            "minimum": (
                min(
                    episode_weight_values
                )
            ),

            "maximum": (
                max(
                    episode_weight_values
                )
            ),

            "mean": float(
                np.mean(
                    episode_weight_values
                )
            ),
        },

        "class_balancing": {
            "method": (
                "inverse_sqrt_frequency"
            ),

            "minimum_weight": (
                MIN_CLASS_WEIGHT
            ),

            "maximum_weight": (
                MAX_CLASS_WEIGHT
            ),
        },

        "training": {
            "epochs": (
                epoch_reports
            ),
        },

        "holdout": (
            holdout
        ),
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.write("\n")

    print()
    print(
        "Training complete."
    )

    print(
        f"Model:  {model_path}"
    )

    print(
        f"Report: {report_path}"
    )

    print(
        "Exact operation sequence accuracy: "
        f"{holdout['exact_operation_sequence_accuracy']:.4f}"
    )

    for slot in holdout["slots"]:
        print(
            f"slot {slot['slot']}: "
            f"orders={slot['real_orders']}, "
            f"op_acc={slot['operation_accuracy']}, "
            f"active_op_acc={slot['active_operation_accuracy']}, "
            f"arg_acc={slot['argument_accuracy']}, "
            f"qty_mae={slot['quantity_mae']}, "
            f"true_active={slot['true_active_rate']}, "
            f"pred_active={slot['predicted_active_rate']}"
        )


if __name__ == "__main__":
    main()