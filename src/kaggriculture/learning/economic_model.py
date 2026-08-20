"""Runtime economic policy for the multi-slot Kaggriculture model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.kaggriculture.core.action_codec import (
    ARGUMENTS,
    MARKET_NO_ARGUMENT_OPERATIONS,
    MARKET_OPERATIONS,
)
from src.kaggriculture.data.feature_extractor import FeatureExtractor


DEFAULT_MODEL_PATH = Path("experiments/behavior_cloning/artifacts/economic_bc.npz")

QUANTITY_OPERATIONS = {
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
}

NO_ORDER_ID = MARKET_OPERATIONS.index("NO_ORDER")


@dataclass
class SlotRuntime:
    operation_coef: np.ndarray
    operation_intercept: np.ndarray

    argument_initialized: bool
    argument_coef: np.ndarray | None
    argument_intercept: np.ndarray | None

    quantity_initialized: bool
    quantity_coef: np.ndarray | None
    quantity_intercept: np.ndarray | None


class EconomicModel:
    """Load and execute the multi-slot economic BC policy."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Economic model not found: {self.model_path}. "
                "Run train_economic_model.py first."
            )

        self.extractor = FeatureExtractor()

        self._load()

    def _load(self) -> None:
        with np.load(
            self.model_path,
            allow_pickle=False,
        ) as data:
            self.version = int(data["version"][0])

            if self.version not in {2, 3}:
                raise ValueError(
                    f"Expected economic model version 2, got {self.version}"
                )

            self.feature_count = int(
                data["feature_count"][0]
            )

            self.max_orders = int(
                data["max_orders"][0]
            )

            saved_operations = tuple(
                str(value)
                for value in data[
                    "market_operations"
                ].tolist()
            )

            saved_arguments = tuple(
                str(value)
                for value in data[
                    "arguments"
                ].tolist()
            )

            quantity_transform = data.get(
                "quantity_transform"
            )

            if quantity_transform is None:
                self.quantity_transform = "identity"
            else:
                self.quantity_transform = str(
                    quantity_transform[0]
                )

            if saved_operations != tuple(
                MARKET_OPERATIONS
            ):
                raise ValueError(
                    "Market operation schema does not "
                    "match the model."
                )

            if saved_arguments != tuple(ARGUMENTS):
                raise ValueError(
                    "Argument schema does not match "
                    "the model."
                )

            self.slots: list[SlotRuntime] = []

            for slot_index in range(
                self.max_orders
            ):
                prefix = f"slot_{slot_index}"

                operation_coef = np.asarray(
                    data[
                        f"{prefix}_operation_coef"
                    ],
                    dtype=np.float32,
                )

                operation_intercept = np.asarray(
                    data[
                        f"{prefix}_operation_intercept"
                    ],
                    dtype=np.float32,
                )

                argument_initialized = bool(
                    data[
                        f"{prefix}_argument_initialized"
                    ][0]
                )

                if argument_initialized:
                    argument_coef = np.asarray(
                        data[
                            f"{prefix}_argument_coef"
                        ],
                        dtype=np.float32,
                    )

                    argument_intercept = np.asarray(
                        data[
                            f"{prefix}_argument_intercept"
                        ],
                        dtype=np.float32,
                    )
                else:
                    argument_coef = None
                    argument_intercept = None

                quantity_initialized = bool(
                    data[
                        f"{prefix}_quantity_initialized"
                    ][0]
                )

                if quantity_initialized:
                    quantity_coef = np.asarray(
                        data[
                            f"{prefix}_quantity_coef"
                        ],
                        dtype=np.float32,
                    ).reshape(
                        self.feature_count
                    )

                    quantity_intercept = np.asarray(
                        data[
                            f"{prefix}_quantity_intercept"
                        ],
                        dtype=np.float32,
                    )
                else:
                    quantity_coef = None
                    quantity_intercept = None

                self.slots.append(
                    SlotRuntime(
                        operation_coef=operation_coef,
                        operation_intercept=operation_intercept,
                        argument_initialized=argument_initialized,
                        argument_coef=argument_coef,
                        argument_intercept=argument_intercept,
                        quantity_initialized=quantity_initialized,
                        quantity_coef=quantity_coef,
                        quantity_intercept=quantity_intercept,
                    )
                )

        if (
            self.extractor.feature_count
            != self.feature_count
        ):
            raise ValueError(
                "Feature schema does not match "
                "the economic model: "
                f"model={self.feature_count}, "
                f"extractor={self.extractor.feature_count}"
            )

        for slot_index, slot in enumerate(
            self.slots
        ):
            expected_operation_shape = (
                len(MARKET_OPERATIONS),
                self.feature_count,
            )

            if (
                slot.operation_coef.shape
                != expected_operation_shape
            ):
                raise ValueError(
                    f"Unexpected operation weights "
                    f"for slot {slot_index}: "
                    f"{slot.operation_coef.shape}"
                )

            if slot.argument_initialized:
                expected_argument_shape = (
                    len(ARGUMENTS),
                    self.feature_count,
                )

                if (
                    slot.argument_coef is None
                    or slot.argument_coef.shape
                    != expected_argument_shape
                ):
                    raise ValueError(
                        f"Unexpected argument weights "
                        f"for slot {slot_index}"
                    )

    @staticmethod
    def _sparse_logits(
        indices: Sequence[int],
        values: Sequence[float],
        weights: np.ndarray,
        bias: np.ndarray,
    ) -> np.ndarray:
        if len(indices) != len(values):
            raise ValueError(
                "Feature indices and values have "
                "different lengths."
            )

        if not indices:
            return np.asarray(
                bias,
                dtype=np.float32,
            ).copy()

        index_array = np.asarray(
            indices,
            dtype=np.int64,
        )

        value_array = np.asarray(
            values,
            dtype=np.float32,
        )

        logits = (
            weights[:, index_array]
            @ value_array
            + bias
        )

        return np.asarray(
            logits,
            dtype=np.float32,
        )

    @staticmethod
    def _sparse_regression(
        indices: Sequence[int],
        values: Sequence[float],
        weights: np.ndarray,
        bias: np.ndarray,
    ) -> float:
        if len(indices) != len(values):
            raise ValueError(
                "Feature indices and values have "
                "different lengths."
            )

        result = float(
            np.asarray(bias).reshape(-1)[0]
        )

        if indices:
            index_array = np.asarray(
                indices,
                dtype=np.int64,
            )

            value_array = np.asarray(
                values,
                dtype=np.float32,
            )

            result += float(
                weights[index_array]
                @ value_array
            )

        return result

    def _predict_operation(
        self,
        slot: SlotRuntime,
        indices: Sequence[int],
        values: Sequence[float],
    ) -> tuple[int, np.ndarray]:
        logits = self._sparse_logits(
            indices,
            values,
            slot.operation_coef,
            slot.operation_intercept,
        )

        operation_id = int(
            np.argmax(logits)
        )

        return operation_id, logits

    def _predict_argument(
        self,
        slot: SlotRuntime,
        indices: Sequence[int],
        values: Sequence[float],
    ) -> tuple[int, np.ndarray | None]:
        if (
            not slot.argument_initialized
            or slot.argument_coef is None
            or slot.argument_intercept is None
        ):
            return 0, None

        logits = self._sparse_logits(
            indices,
            values,
            slot.argument_coef,
            slot.argument_intercept,
        )

        argument_id = int(
            np.argmax(logits)
        )

        return argument_id, logits

    def _predict_quantity(
        self,
        slot: SlotRuntime,
        indices: Sequence[int],
        values: Sequence[float],
    ) -> int:
        if (
            not slot.quantity_initialized
            or slot.quantity_coef is None
            or slot.quantity_intercept is None
        ):
            return 1

        raw = self._sparse_regression(
            indices,
            values,
            slot.quantity_coef,
            slot.quantity_intercept,
        )

        if self.quantity_transform == "log1p":
            raw = float(
                np.expm1(raw)
            )

        if not np.isfinite(raw):
            return 1

        return int(
            np.clip(
                np.rint(raw),
                1,
                100,
            )
        )

    def predict_slot(
        self,
        observation: Any,
        slot_index: int,
    ) -> dict[str, Any]:
        if not 0 <= slot_index < self.max_orders:
            raise IndexError(slot_index)

        features = self.extractor.extract(
            observation
        )

        return self._predict_slot_from_features(
            features.indices,
            features.values,
            slot_index,
        )

    def _predict_slot_from_features(
        self,
        indices: Sequence[int],
        values: Sequence[float],
        slot_index: int,
    ) -> dict[str, Any]:
        slot = self.slots[slot_index]

        (
            operation_id,
            operation_logits,
        ) = self._predict_operation(
            slot,
            indices,
            values,
        )

        operation = MARKET_OPERATIONS[
            operation_id
        ]

        if (
            operation == "NO_ORDER"
            or operation
            in MARKET_NO_ARGUMENT_OPERATIONS
        ):
            argument_id = 0
            argument_logits = None
        else:
            (
                argument_id,
                argument_logits,
            ) = self._predict_argument(
                slot,
                indices,
                values,
            )

        argument = ARGUMENTS[
            argument_id
        ]

        if operation in QUANTITY_OPERATIONS:
            quantity = self._predict_quantity(
                slot,
                indices,
                values,
            )
        else:
            quantity = 0

        return {
            "slot": slot_index,
            "operation_id": operation_id,
            "operation": operation,
            "argument_id": argument_id,
            "argument": argument,
            "quantity": quantity,
            "operation_logits": operation_logits,
            "argument_logits": argument_logits,
        }

    def predict_encoded(
        self,
        observation: Any,
    ) -> list[dict[str, Any]]:
        """
        Predict the complete ordered market sequence.

        Prediction stops at the first NO_ORDER.
        """

        features = self.extractor.extract(
            observation
        )

        predictions: list[
            dict[str, Any]
        ] = []

        for slot_index in range(
            self.max_orders
        ):
            prediction = (
                self._predict_slot_from_features(
                    features.indices,
                    features.values,
                    slot_index,
                )
            )

            predictions.append(
                prediction
            )

            if (
                prediction["operation"]
                == "NO_ORDER"
            ):
                break

        return predictions

    def _decode_prediction(
        self,
        prediction: dict[str, Any],
    ) -> list[Any] | None:
        operation = prediction[
            "operation"
        ]

        if operation == "NO_ORDER":
            return None

        if (
            operation
            in MARKET_NO_ARGUMENT_OPERATIONS
        ):
            return [operation]

        argument = prediction[
            "argument"
        ]

        if operation in QUANTITY_OPERATIONS:
            return [
                operation,
                argument,
                prediction["quantity"],
            ]

        return [
            operation,
            argument,
        ]

    def predict_market(
        self,
        observation: Any,
    ) -> list[list[Any]]:
        """
        Predict up to max_orders ordered market commands.
        """

        encoded = self.predict_encoded(
            observation
        )

        orders: list[list[Any]] = []

        for prediction in encoded:
            command = self._decode_prediction(
                prediction
            )

            if command is None:
                break

            orders.append(command)

        return orders


_default_model: EconomicModel | None = None


def get_economic_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> EconomicModel:
    """Lazily load the economic model."""

    global _default_model

    requested_path = Path(
        model_path
    )

    if (
        _default_model is None
        or _default_model.model_path
        != requested_path
    ):
        _default_model = EconomicModel(
            requested_path
        )

    return _default_model


def predict_market(
    observation: Any,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> list[list[Any]]:
    """Convenience function for agent code."""

    return get_economic_model(
        model_path
    ).predict_market(
        observation
    )