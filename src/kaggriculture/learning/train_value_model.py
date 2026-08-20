"""Train an economic state-action value model with state-action interactions."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from src.kaggriculture.learning.economic_value_features import (
    TOTAL_FEATURE_COUNT,
    build_from_encoded,
)


DEFAULT_DATASET = "data/processed/economic_value_dataset.jsonl.gz"
DEFAULT_MODEL = "experiments/behavior_cloning/artifacts/economic_value.npz"
DEFAULT_REPORT = "experiments/behavior_cloning/artifacts/economic_value_report.json"

TARGET_SCALE = 1000.0

EPOCHS = 20
LEARNING_RATE = 0.01
L2 = 2e-5

RANDOM_SEED = 17


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(
            path,
            "rt",
            encoding="utf-8",
        )

    return path.open(
        "r",
        encoding="utf-8",
    )


def load_dataset(
    path: Path,
):
    train = []
    holdout = []

    with open_text(path) as source:
        for line_number, line in enumerate(
            source,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                row = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            try:
                (
                    indices,
                    values,
                ) = build_from_encoded(
                    state_indices=row[
                        "feature_indices"
                    ],
                    state_values=row[
                        "feature_values"
                    ],
                    operation_ids=row[
                        "action_operation_ids"
                    ],
                    argument_ids=row[
                        "action_argument_ids"
                    ],
                    quantities=row[
                        "action_quantities"
                    ],
                    active=row[
                        "action_active"
                    ],
                )

                margin = float(
                    row["target_margin"]
                )

                episode_id = int(
                    row["episode_id"]
                )

                split = str(
                    row["split"]
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"Invalid value-dataset record "
                    f"on line {line_number}: {exc}"
                ) from exc

            if not math.isfinite(
                margin
            ):
                margin = 0.0

            target = float(
                np.arcsinh(
                    margin
                    / TARGET_SCALE
                )
            )

            example = (
                indices,
                values,
                target,
                margin,
                episode_id,
            )

            if split == "train":
                train.append(
                    example
                )

            elif split == "holdout":
                holdout.append(
                    example
                )

            else:
                raise ValueError(
                    f"Unknown split {split!r} "
                    f"on line {line_number}"
                )

    if not train:
        raise RuntimeError(
            "No train examples found"
        )

    if not holdout:
        raise RuntimeError(
            "No holdout examples found"
        )

    return train, holdout


class LinearValueModel:
    def __init__(
        self,
        feature_count: int,
    ):
        self.feature_count = (
            feature_count
        )

        self.weights = np.zeros(
            feature_count,
            dtype=np.float32,
        )

        self.bias = 0.0

    def predict(
        self,
        indices: np.ndarray,
        values: np.ndarray,
    ) -> float:

        return float(
            self.bias
            + np.dot(
                self.weights[
                    indices
                ],
                values,
            )
        )

    def update(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        target: float,
        learning_rate: float,
        l2: float,
    ) -> float:

        prediction = self.predict(
            indices,
            values,
        )

        error = (
            prediction
            - target
        )

        # Robust clipped gradient.
        gradient = float(
            np.clip(
                error,
                -3.0,
                3.0,
            )
        )

        current = self.weights[
            indices
        ].copy()

        self.weights[
            indices
        ] -= learning_rate * (
            gradient * values
            + l2 * current
        )

        self.bias -= (
            learning_rate
            * gradient
        )

        return (
            error * error
        )


def transformed_to_margin(
    prediction: float,
) -> float:

    prediction = float(
        np.clip(
            prediction,
            -10.0,
            10.0,
        )
    )

    return float(
        np.sinh(
            prediction
        )
        * TARGET_SCALE
    )


def safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    if len(x) < 2:
        return 0.0

    if (
        np.std(x) <= 0
        or np.std(y) <= 0
    ):
        return 0.0

    value = float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )

    if not math.isfinite(
        value
    ):
        return 0.0

    return value


def evaluate(
    model: LinearValueModel,
    examples,
):
    transformed_squared_errors = []
    margin_absolute_errors = []

    predicted_margins = []
    true_margins = []

    episode_predictions = {}
    episode_targets = {}

    for (
        indices,
        values,
        target,
        margin,
        episode_id,
    ) in examples:

        transformed_prediction = (
            model.predict(
                indices,
                values,
            )
        )

        predicted_margin = (
            transformed_to_margin(
                transformed_prediction
            )
        )

        transformed_squared_errors.append(
            (
                transformed_prediction
                - target
            )
            ** 2
        )

        margin_absolute_errors.append(
            abs(
                predicted_margin
                - margin
            )
        )

        predicted_margins.append(
            predicted_margin
        )

        true_margins.append(
            margin
        )

        episode_predictions.setdefault(
            episode_id,
            [],
        ).append(
            predicted_margin
        )

        episode_targets.setdefault(
            episode_id,
            [],
        ).append(
            margin
        )

    predicted_array = np.asarray(
        predicted_margins,
        dtype=np.float64,
    )

    target_array = np.asarray(
        true_margins,
        dtype=np.float64,
    )

    margin_correlation = (
        safe_correlation(
            predicted_array,
            target_array,
        )
    )

    episode_pred = []
    episode_true = []

    for episode_id in sorted(
        episode_predictions
    ):
        episode_pred.append(
            float(
                np.mean(
                    episode_predictions[
                        episode_id
                    ]
                )
            )
        )

        episode_true.append(
            float(
                np.mean(
                    episode_targets[
                        episode_id
                    ]
                )
            )
        )

    episode_correlation = (
        safe_correlation(
            np.asarray(
                episode_pred,
                dtype=np.float64,
            ),
            np.asarray(
                episode_true,
                dtype=np.float64,
            ),
        )
    )

    return {
        "examples": len(
            examples
        ),

        "transformed_rmse": float(
            np.sqrt(
                np.mean(
                    transformed_squared_errors
                )
            )
        ),

        "margin_mae": float(
            np.mean(
                margin_absolute_errors
            )
        ),

        "margin_median_ae": float(
            np.median(
                margin_absolute_errors
            )
        ),

        "margin_correlation": (
            margin_correlation
        ),

        "episode_correlation": (
            episode_correlation
        ),
    }


def train(
    dataset_path: Path,
    model_path: Path,
    report_path: Path,
):
    print(
        "Loading economic value dataset..."
    )

    (
        train_examples,
        holdout_examples,
    ) = load_dataset(
        dataset_path
    )

    print(
        f"Train examples:   "
        f"{len(train_examples)}"
    )

    print(
        f"Holdout examples: "
        f"{len(holdout_examples)}"
    )

    print()
    print(
        "Training economic "
        "state-action value model"
    )

    print(
        f"features: {TOTAL_FEATURE_COUNT}"
    )

    print(
        f"epochs:   {EPOCHS}"
    )

    print(
        f"lr:       {LEARNING_RATE}"
    )

    print(
        f"target:   "
        f"asinh(margin / "
        f"{TARGET_SCALE})"
    )

    print(
        "checkpoint metric: "
        "holdout margin correlation"
    )

    model = LinearValueModel(
        TOTAL_FEATURE_COUNT
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    order = np.arange(
        len(
            train_examples
        )
    )

    best_correlation = -float(
        "inf"
    )

    best_episode_correlation = (
        -float(
            "inf"
        )
    )

    best_epoch = 0

    best_weights = (
        model.weights.copy()
    )

    best_bias = float(
        model.bias
    )

    best_holdout_metrics = None

    epoch_reports = []

    for epoch_index in range(
        EPOCHS
    ):
        epoch_number = (
            epoch_index + 1
        )

        rng.shuffle(
            order
        )

        losses = []

        learning_rate = (
            LEARNING_RATE
            / math.sqrt(
                epoch_number
            )
        )

        for example_index in order:
            (
                indices,
                values,
                target,
                _,
                _,
            ) = train_examples[
                example_index
            ]

            loss = model.update(
                indices=indices,
                values=values,
                target=target,
                learning_rate=(
                    learning_rate
                ),
                l2=L2,
            )

            losses.append(
                loss
            )

        train_rmse = float(
            np.sqrt(
                np.mean(
                    losses
                )
            )
        )

        holdout_metrics = (
            evaluate(
                model,
                holdout_examples,
            )
        )

        correlation = float(
            holdout_metrics[
                "margin_correlation"
            ]
        )

        episode_correlation = float(
            holdout_metrics[
                "episode_correlation"
            ]
        )

        # Primary metric:
        # state-action ranking correlation.
        #
        # Episode correlation is only a
        # tie-breaker.
        is_best = (
            correlation
            > best_correlation
            or (
                math.isclose(
                    correlation,
                    best_correlation,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                and episode_correlation
                > best_episode_correlation
            )
        )

        if is_best:
            best_correlation = (
                correlation
            )

            best_episode_correlation = (
                episode_correlation
            )

            best_epoch = (
                epoch_number
            )

            best_weights = (
                model.weights.copy()
            )

            best_bias = float(
                model.bias
            )

            best_holdout_metrics = (
                dict(
                    holdout_metrics
                )
            )

        epoch_reports.append(
            {
                "epoch": (
                    epoch_number
                ),

                "learning_rate": float(
                    learning_rate
                ),

                "train_rmse": (
                    train_rmse
                ),

                "holdout": dict(
                    holdout_metrics
                ),

                "is_best": (
                    is_best
                ),
            }
        )

        marker = (
            "  <-- BEST"
            if is_best
            else ""
        )

        print(
            f"epoch "
            f"{epoch_number:2d}/"
            f"{EPOCHS}: "
            f"train_rmse="
            f"{train_rmse:.4f} | "
            f"holdout_rmse="
            f"{holdout_metrics['transformed_rmse']:.4f} | "
            f"corr="
            f"{correlation:.4f} | "
            f"episode_corr="
            f"{episode_correlation:.4f}"
            f"{marker}"
        )

    # ---------------------------------------------------------
    # Restore best checkpoint.
    # ---------------------------------------------------------

    model.weights[:] = (
        best_weights
    )

    model.bias = (
        best_bias
    )

    print()
    print(
        "Restored best checkpoint"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        "Best holdout correlation: "
        f"{best_correlation:.4f}"
    )

    print(
        "Best episode correlation: "
        f"{best_episode_correlation:.4f}"
    )

    print()
    print(
        "Final evaluation..."
    )

    train_metrics = evaluate(
        model,
        train_examples,
    )

    holdout_metrics = evaluate(
        model,
        holdout_examples,
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        model_path,

        version=np.asarray(
            [3],
            dtype=np.int32,
        ),

        weights=np.asarray(
            model.weights,
            dtype=np.float32,
        ),

        bias=np.asarray(
            [model.bias],
            dtype=np.float32,
        ),

        total_feature_count=np.asarray(
            [
                TOTAL_FEATURE_COUNT
            ],
            dtype=np.int32,
        ),

        target_scale=np.asarray(
            [
                TARGET_SCALE
            ],
            dtype=np.float32,
        ),

        max_orders=np.asarray(
            [10],
            dtype=np.int32,
        ),

        best_epoch=np.asarray(
            [best_epoch],
            dtype=np.int32,
        ),

        best_holdout_correlation=(
            np.asarray(
                [
                    best_correlation
                ],
                dtype=np.float32,
            )
        ),

        best_episode_correlation=(
            np.asarray(
                [
                    best_episode_correlation
                ],
                dtype=np.float32,
            )
        ),
    )

    report = {
        "version": 3,

        "architecture": (
            "linear_state_action_"
            "value_with_hashed_"
            "interactions"
        ),

        "dataset": str(
            dataset_path
        ),

        "model": str(
            model_path
        ),

        "total_feature_count": (
            TOTAL_FEATURE_COUNT
        ),

        "target_scale": (
            TARGET_SCALE
        ),

        "epochs": (
            EPOCHS
        ),

        "learning_rate": (
            LEARNING_RATE
        ),

        "l2": (
            L2
        ),

        "checkpoint_selection": {
            "metric": (
                "holdout_margin_correlation"
            ),

            "tie_breaker": (
                "holdout_episode_correlation"
            ),

            "best_epoch": (
                best_epoch
            ),

            "best_holdout_correlation": (
                best_correlation
            ),

            "best_episode_correlation": (
                best_episode_correlation
            ),

            "best_holdout_metrics": (
                best_holdout_metrics
            ),
        },

        "epoch_history": (
            epoch_reports
        ),

        "train": (
            train_metrics
        ),

        "holdout": (
            holdout_metrics
        ),
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as output:

        json.dump(
            report,
            output,
            ensure_ascii=False,
            indent=2,
        )

        output.write(
            "\n"
        )

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

    print()
    print(
        "Selected checkpoint:"
    )

    print(
        f"  epoch: "
        f"{best_epoch}"
    )

    print(
        "  holdout correlation: "
        f"{best_correlation:.4f}"
    )

    print(
        "  episode correlation: "
        f"{best_episode_correlation:.4f}"
    )

    print()
    print(
        "Holdout:"
    )

    for (
        key,
        value,
    ) in holdout_metrics.items():

        if isinstance(
            value,
            float,
        ):
            print(
                f"  {key}: "
                f"{value:.4f}"
            )

        else:
            print(
                f"  {key}: "
                f"{value}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT,
    )

    args = parser.parse_args()

    train(
        dataset_path=Path(
            args.dataset
        ),

        model_path=Path(
            args.model
        ),

        report_path=Path(
            args.report
        ),
    )


if __name__ == "__main__":
    main()