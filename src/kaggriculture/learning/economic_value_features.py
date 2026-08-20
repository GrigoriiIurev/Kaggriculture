"""Shared state-action feature encoding for the economic value model."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from src.kaggriculture.core.action_codec import ActionEncoder
from src.kaggriculture.data.feature_extractor import FeatureExtractor


EXTRACTOR = FeatureExtractor()

STATE_FEATURE_COUNT = EXTRACTOR.feature_count
SCALAR_FEATURE_COUNT = EXTRACTOR.scalar_count

MAX_ORDERS = 10

ACTION_FEATURE_COUNT = 2048
INTERACTION_FEATURE_COUNT = 8192

ACTION_OFFSET = STATE_FEATURE_COUNT
INTERACTION_OFFSET = (
    STATE_FEATURE_COUNT
    + ACTION_FEATURE_COUNT
)

TOTAL_FEATURE_COUNT = (
    STATE_FEATURE_COUNT
    + ACTION_FEATURE_COUNT
    + INTERACTION_FEATURE_COUNT
)

ENCODER = ActionEncoder()


def stable_hash(*values: Any) -> int:
    """Deterministic FNV-1a hash."""

    h = 2166136261

    text = "|".join(
        str(value)
        for value in values
    )

    for byte in text.encode("utf-8"):
        h ^= byte
        h *= 16777619
        h &= 0xFFFFFFFF

    return h


def _add(
    features: dict[int, float],
    index: int,
    value: float,
) -> None:
    if value == 0:
        return

    features[index] = (
        features.get(index, 0.0)
        + float(value)
    )


def build_from_encoded(
    state_indices: Sequence[int],
    state_values: Sequence[float],
    operation_ids: Sequence[int],
    argument_ids: Sequence[int],
    quantities: Sequence[float],
    active: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build value-model features from an already encoded state/action.

    Besides ordinary state and action features, this adds interactions
    between important scalar state features and each market command.

    Those interactions are what allow the model to learn things such as:

        HIRE is different when money=3000 vs money=20;
        BUY_LAND is different early vs late in the season;
        BUY_PRODUCT WHEAT depends on shed stock and animal count.
    """

    features: dict[int, float] = {}

    # ---------------------------------------------------------
    # State
    # ---------------------------------------------------------

    scalar_state: list[
        tuple[int, float]
    ] = []

    for index, value in zip(
        state_indices,
        state_values,
    ):
        index = int(index)
        value = float(value)

        if not 0 <= index < STATE_FEATURE_COUNT:
            raise ValueError(
                f"State feature index {index} "
                f"outside [0, {STATE_FEATURE_COUNT})"
            )

        _add(
            features,
            index,
            value,
        )

        # Economic interactions only need the compact global
        # scalar summary, not every board cell.
        if index < SCALAR_FEATURE_COUNT:
            scalar_state.append(
                (index, value)
            )

    # ---------------------------------------------------------
    # Action
    # ---------------------------------------------------------

    active_count = 0

    for slot in range(MAX_ORDERS):
        if (
            slot >= len(active)
            or not active[slot]
        ):
            continue

        active_count += 1

        operation = int(
            operation_ids[slot]
        )

        argument = int(
            argument_ids[slot]
        )

        quantity = max(
            0.0,
            float(
                quantities[slot]
            ),
        )

        quantity_log = math.log1p(
            quantity
        )

        # -----------------------------------------------------
        # Pure action features
        # -----------------------------------------------------

        action_index = stable_hash(
            "slot_op",
            slot,
            operation,
        ) % ACTION_FEATURE_COUNT

        _add(
            features,
            ACTION_OFFSET + action_index,
            1.0,
        )

        action_index = stable_hash(
            "slot_op_arg",
            slot,
            operation,
            argument,
        ) % ACTION_FEATURE_COUNT

        _add(
            features,
            ACTION_OFFSET + action_index,
            1.0,
        )

        action_index = stable_hash(
            "slot_quantity",
            slot,
            operation,
            argument,
        ) % ACTION_FEATURE_COUNT

        _add(
            features,
            ACTION_OFFSET + action_index,
            quantity_log,
        )

        # -----------------------------------------------------
        # State × action interactions
        # -----------------------------------------------------

        for (
            state_index,
            state_value,
        ) in scalar_state:

            interaction_index = (
                stable_hash(
                    "state_action",
                    state_index,
                    slot,
                    operation,
                    argument,
                )
                % INTERACTION_FEATURE_COUNT
            )

            _add(
                features,
                (
                    INTERACTION_OFFSET
                    + interaction_index
                ),
                state_value,
            )

            if quantity > 0:
                interaction_index = (
                    stable_hash(
                        "state_action_quantity",
                        state_index,
                        slot,
                        operation,
                        argument,
                    )
                    % INTERACTION_FEATURE_COUNT
                )

                _add(
                    features,
                    (
                        INTERACTION_OFFSET
                        + interaction_index
                    ),
                    (
                        state_value
                        * quantity_log
                    ),
                )

    # ---------------------------------------------------------
    # Number of orders
    # ---------------------------------------------------------

    count_index = stable_hash(
        "order_count",
        active_count,
    ) % ACTION_FEATURE_COUNT

    _add(
        features,
        ACTION_OFFSET + count_index,
        1.0,
    )

    # Also interact order count with scalar state.
    for (
        state_index,
        state_value,
    ) in scalar_state:

        interaction_index = (
            stable_hash(
                "state_order_count",
                state_index,
                active_count,
            )
            % INTERACTION_FEATURE_COUNT
        )

        _add(
            features,
            (
                INTERACTION_OFFSET
                + interaction_index
            ),
            state_value,
        )

    ordered = sorted(
        features.items()
    )

    return (
        np.asarray(
            [
                index
                for index, _
                in ordered
            ],
            dtype=np.int32,
        ),
        np.asarray(
            [
                value
                for _, value
                in ordered
            ],
            dtype=np.float32,
        ),
    )


def encode_market(
    market: Sequence[
        Sequence[Any]
    ],
) -> tuple[
    list[int],
    list[int],
    list[float],
    list[int],
]:
    """Encode normal Kaggriculture market commands."""

    operation_ids = []
    argument_ids = []
    quantities = []
    active = []

    market = list(
        market
    )[:MAX_ORDERS]

    for slot in range(
        MAX_ORDERS
    ):
        if slot >= len(market):
            operation_ids.append(0)
            argument_ids.append(0)
            quantities.append(0.0)
            active.append(0)
            continue

        encoded = ENCODER.encode_market(
            market[slot]
        )

        operation_ids.append(
            encoded.operation_id
        )

        argument_ids.append(
            encoded.argument_id
        )

        quantities.append(
            float(
                encoded.quantity
            )
        )

        active.append(1)

    return (
        operation_ids,
        argument_ids,
        quantities,
        active,
    )


def build_from_observation(
    observation: Any,
    market: Sequence[
        Sequence[Any]
    ],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runtime version used by the actual agent.
    """

    state = EXTRACTOR.extract(
        observation
    )

    (
        operation_ids,
        argument_ids,
        quantities,
        active,
    ) = encode_market(
        market
    )

    return build_from_encoded(
        state_indices=state.indices,
        state_values=state.values,
        operation_ids=operation_ids,
        argument_ids=argument_ids,
        quantities=quantities,
        active=active,
    )