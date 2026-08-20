"""Lightweight inference for the exported behavior-cloning worker model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.kaggriculture.core.action_codec import (
    ARGUMENTS,
    ARGUMENT_TO_ID,
    ActionDecoder,
    WORKER_OPERATIONS,
)
from src.kaggriculture.core.legal_actions import (
    ARGUMENT_OPERATIONS,
    legal_worker_arguments,
    legal_worker_operations,
)
from src.kaggriculture.core.state_parser import parse_observation
from src.kaggriculture.data.worker_dataset import WorkerFeatureExtractor


class BehaviorCloningPolicy:
    def __init__(self, model_path: str | Path, board_size: int = 10):
        model = np.load(model_path, allow_pickle=False)
        self.feature_count = int(model["feature_count"][0])
        self.operation_classes = model["operation_classes"].astype(np.int16)
        self.operation_coef = model["operation_coef"].astype(np.float32)
        self.operation_intercept = model["operation_intercept"].astype(np.float32)
        self.argument_classes = model["argument_classes"].astype(np.int16)
        self.argument_coef = model["argument_coef"].astype(np.float32)
        self.argument_intercept = model["argument_intercept"].astype(np.float32)
        self.argument_feature_count = int(model["argument_feature_count"][0])
        self.extractor = WorkerFeatureExtractor(board_size=board_size)
        self.decoder = ActionDecoder()
        if self.feature_count != self.extractor.feature_count:
            raise ValueError(
                f"Model expects {self.feature_count} features, "
                f"extractor produces {self.extractor.feature_count}"
            )
        expected_argument_features = self.feature_count + len(WORKER_OPERATIONS)
        if self.argument_feature_count != expected_argument_features:
            raise ValueError(
                f"Argument model expects {self.argument_feature_count} features, "
                f"expected {expected_argument_features}"
            )

    def predict_commands(self, observation: Any) -> tuple[list[Any], ...]:
        state = parse_observation(observation)
        base = self.extractor.base.extract(observation)
        commands = []
        for worker_index in range(len(state.units)):
            context = self.extractor.context(state, worker_index)
            indices = np.fromiter(
                (*base.indices, *context.indices), dtype=np.int32
            )
            values = np.fromiter(
                (*base.values, *context.values), dtype=np.float32
            )
            operation_scores = self._scores(
                self.operation_classes,
                self.operation_coef,
                self.operation_intercept,
                indices,
                values,
            )
            legal_operations = legal_worker_operations(state, worker_index)
            operation_id = self._best_legal_operation(
                operation_scores,
                legal_operations,
                state,
                worker_index,
            )
            argument_scores = self._scores(
                self.argument_classes,
                self.argument_coef,
                self.argument_intercept,
                np.append(indices, self.feature_count + operation_id),
                np.append(values, np.float32(1)),
            )
            argument_id = self._best_legal_argument(
                argument_scores, state, worker_index, operation_id
            )
            commands.append(
                self.decoder.decode_worker(operation_id, argument_id, quantity=1)
            )
        return tuple(commands)

    @staticmethod
    def _scores(
        classes: np.ndarray,
        coefficients: np.ndarray,
        intercept: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
    ) -> dict[int, float]:
        scores = intercept + coefficients[:, indices] @ values
        return {
            int(action_class): float(score)
            for action_class, score in zip(classes, scores)
        }

    def _best_legal_operation(
        self,
        scores: dict[int, float],
        legal_operations: set[str],
        state,
        worker_index: int,
    ) -> int:
        candidates = []
        for operation_id, score in scores.items():
            operation = WORKER_OPERATIONS[operation_id]
            if operation not in legal_operations:
                continue
            if operation in ARGUMENT_OPERATIONS:
                legal_arguments = legal_worker_arguments(
                    state, worker_index, operation
                )
                learned_arguments = {
                    ARGUMENTS[argument_id] for argument_id in self.argument_classes
                }
                if not legal_arguments.intersection(learned_arguments):
                    continue
            candidates.append((score, operation_id))
        if not candidates:
            return WORKER_OPERATIONS.index("PASS")
        return max(candidates)[1]

    def _best_legal_argument(
        self,
        scores: dict[int, float],
        state,
        worker_index: int,
        operation_id: int,
    ) -> int:
        operation = WORKER_OPERATIONS[operation_id]
        if operation not in ARGUMENT_OPERATIONS:
            return ARGUMENT_TO_ID["NONE"]
        legal_arguments = legal_worker_arguments(state, worker_index, operation)
        candidates = [
            (score, argument_id)
            for argument_id, score in scores.items()
            if ARGUMENTS[argument_id] in legal_arguments
        ]
        return max(candidates)[1] if candidates else ARGUMENT_TO_ID["NONE"]


def commands_to_action(
    commands: tuple[list[Any], ...], market: list[list[Any]] | None = None
) -> dict[str, Any]:
    if not commands:
        raise ValueError("At least the main farmer command is required")
    return {
        "farmer": commands[0],
        "hands": list(commands[1:]),
        "market": market or [],
    }
