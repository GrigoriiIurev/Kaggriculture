"""Build state-action value-learning records for the economic policy."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from src.kaggriculture.core.action_codec import (
    ARGUMENTS,
    MARKET_OPERATIONS,
)


DEFAULT_INPUT = "data/processed/economic_dataset.jsonl.gz"
DEFAULT_OUTPUT = "data/processed/economic_value_dataset.jsonl.gz"
DEFAULT_SCHEMA = "data/processed/economic_value_schema.json"
DEFAULT_MANIFEST = "data/processed/economic_value_manifest.json"

MAX_ORDERS = 10

NO_ORDER_ID = MARKET_OPERATIONS.index("NO_ORDER")
NONE_ARGUMENT_ID = ARGUMENTS.index("NONE")


def _open_text(
    path: Path,
    mode: str,
) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(
            path,
            mode,
            encoding="utf-8",
            compresslevel=6,
        )

    return path.open(
        mode,
        encoding="utf-8",
    )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(result):
        return default

    return result


def _encode_action_sequence(
    targets: list[dict[str, Any]],
) -> dict[str, list]:
    """
    Convert up to 10 ordered market commands into fixed-size arrays.

    Missing slots are explicit NO_ORDER commands.
    """

    operation_ids: list[int] = []
    argument_ids: list[int] = []
    quantities: list[float] = []
    active: list[int] = []

    for slot in range(MAX_ORDERS):
        if slot < len(targets):
            target = targets[slot]

            operation_id = int(
                target.get(
                    "operation_id",
                    NO_ORDER_ID,
                )
            )

            argument_id = int(
                target.get(
                    "argument_id",
                    NONE_ARGUMENT_ID,
                )
            )

            quantity = _safe_float(
                target.get(
                    "quantity",
                    0,
                )
            )

        else:
            operation_id = NO_ORDER_ID
            argument_id = NONE_ARGUMENT_ID
            quantity = 0.0

        if not (
            0
            <= operation_id
            < len(MARKET_OPERATIONS)
        ):
            raise ValueError(
                f"Invalid market operation id: "
                f"{operation_id}"
            )

        if not (
            0
            <= argument_id
            < len(ARGUMENTS)
        ):
            raise ValueError(
                f"Invalid argument id: "
                f"{argument_id}"
            )

        is_active = int(
            operation_id != NO_ORDER_ID
        )

        operation_ids.append(
            operation_id
        )

        argument_ids.append(
            argument_id
        )

        quantities.append(
            float(quantity)
        )

        active.append(
            is_active
        )

    return {
        "operation_ids": operation_ids,
        "argument_ids": argument_ids,
        "quantities": quantities,
        "active": active,
    }


def _outcome_target(
    outcome: Any,
) -> float:
    """
    Numerical auxiliary target:

        loss = -1
        tie  =  0
        win  = +1
    """

    name = str(
        outcome or ""
    ).upper()

    if name in {
        "WIN",
        "WON",
        "VICTORY",
    }:
        return 1.0

    if name in {
        "LOSS",
        "LOSE",
        "LOST",
        "DEFEAT",
    }:
        return -1.0

    return 0.0


def build_value_dataset(
    input_path: str | Path,
    output_path: str | Path,
    schema_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:

    input_path = Path(
        input_path
    )

    output_path = Path(
        output_path
    )

    if schema_path is None:
        schema_path = output_path.with_name(
            "economic_value_schema.json"
        )
    else:
        schema_path = Path(
            schema_path
        )

    if manifest_path is None:
        manifest_path = output_path.with_name(
            "economic_value_manifest.json"
        )
    else:
        manifest_path = Path(
            manifest_path
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record_count = 0

    split_counts = Counter()
    outcome_counts = Counter()
    active_order_counts = Counter()

    episode_ids: set[int] = set()

    margins = []
    rewards = []

    with _open_text(
        input_path,
        "rt",
    ) as source, _open_text(
        output_path,
        "wt",
    ) as output:

        for (
            line_number,
            line,
        ) in enumerate(
            source,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                row = json.loads(
                    line
                )

                episode_id = int(
                    row["episode_id"]
                )

                split = str(
                    row["split"]
                )

                feature_indices = list(
                    row[
                        "feature_indices"
                    ]
                )

                feature_values = list(
                    row[
                        "feature_values"
                    ]
                )

                targets = row.get(
                    "targets",
                    [],
                )

                if not isinstance(
                    targets,
                    list,
                ):
                    raise ValueError(
                        "targets must be a list"
                    )

                action = (
                    _encode_action_sequence(
                        targets
                    )
                )

                margin = _safe_float(
                    row.get(
                        "margin",
                        0.0,
                    )
                )

                final_reward = _safe_float(
                    row.get(
                        "final_reward",
                        0.0,
                    )
                )

                outcome = row.get(
                    "outcome"
                )

            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                raise ValueError(
                    f"Invalid economic dataset "
                    f"record on line "
                    f"{line_number}: {exc}"
                ) from exc

            if len(
                feature_indices
            ) != len(
                feature_values
            ):
                raise ValueError(
                    f"Feature indices/values "
                    f"length mismatch on "
                    f"line {line_number}"
                )

            active_orders = sum(
                action["active"]
            )

            record = {
                "episode_id": episode_id,
                "episode_type": row.get(
                    "episode_type"
                ),
                "seat": row.get(
                    "seat"
                ),
                "step": row.get(
                    "step"
                ),
                "split": split,

                # State.
                "feature_indices": (
                    feature_indices
                ),
                "feature_values": (
                    feature_values
                ),

                # Action actually taken.
                "action_operation_ids": (
                    action[
                        "operation_ids"
                    ]
                ),
                "action_argument_ids": (
                    action[
                        "argument_ids"
                    ]
                ),
                "action_quantities": (
                    action[
                        "quantities"
                    ]
                ),
                "action_active": (
                    action[
                        "active"
                    ]
                ),
                "action_order_count": (
                    active_orders
                ),

                # Value targets.
                #
                # These are NOT inference features.
                "target_margin": margin,
                "target_final_reward": (
                    final_reward
                ),
                "target_outcome": (
                    _outcome_target(
                        outcome
                    )
                ),

                "outcome": outcome,
            }

            json.dump(
                record,
                output,
                ensure_ascii=True,
                separators=(",", ":"),
            )

            output.write("\n")

            record_count += 1

            episode_ids.add(
                episode_id
            )

            split_counts[
                split
            ] += 1

            outcome_counts[
                str(outcome)
            ] += 1

            active_order_counts[
                active_orders
            ] += 1

            margins.append(
                margin
            )

            rewards.append(
                final_reward
            )

    schema = {
        "version": 1,
        "dataset": (
            "economic_state_action_value"
        ),

        "state": {
            "representation": (
                "sparse global features"
            ),
            "schema": (
                "feature_schema.json"
            ),
        },

        "action": {
            "representation": (
                "fixed 10-slot ordered "
                "market sequence"
            ),
            "max_orders": MAX_ORDERS,
            "operations": list(
                MARKET_OPERATIONS
            ),
            "arguments": list(
                ARGUMENTS
            ),
            "fields": [
                "action_operation_ids",
                "action_argument_ids",
                "action_quantities",
                "action_active",
            ],
        },

        "targets": {
            "target_margin": (
                "final player reward "
                "minus opponent reward"
            ),
            "target_final_reward": (
                "final reward of this player"
            ),
            "target_outcome": (
                "-1 loss, 0 tie, +1 win"
            ),
        },

        "important": [
            (
                "Target fields describe the "
                "future and must never be "
                "used as inference inputs."
            ),
            (
                "The dataset contains only "
                "actions actually observed "
                "in replay. It does not "
                "contain counterfactual "
                "rewards for actions that "
                "were not taken."
            ),
            (
                "Train/holdout splitting is "
                "inherited from the economic "
                "dataset and therefore stays "
                "episode-level."
            ),
        ],
    }

    with Path(
        schema_path
    ).open(
        "w",
        encoding="utf-8",
    ) as output:

        json.dump(
            schema,
            output,
            ensure_ascii=False,
            indent=2,
        )

        output.write("\n")

    manifest = {
        "version": 1,

        "input": str(
            input_path
        ),

        "output": str(
            output_path
        ),

        "records": record_count,

        "episodes": len(
            episode_ids
        ),

        "split_counts": dict(
            sorted(
                split_counts.items()
            )
        ),

        "outcome_counts": dict(
            sorted(
                outcome_counts.items()
            )
        ),

        "active_orders_per_turn": {
            str(key): value
            for (
                key,
                value,
            ) in sorted(
                active_order_counts.items()
            )
        },

        "margin": {
            "minimum": (
                min(margins)
                if margins
                else None
            ),
            "mean": (
                float(
                    np.mean(
                        margins
                    )
                )
                if margins
                else None
            ),
            "maximum": (
                max(margins)
                if margins
                else None
            ),
        },

        "final_reward": {
            "minimum": (
                min(rewards)
                if rewards
                else None
            ),
            "mean": (
                float(
                    np.mean(
                        rewards
                    )
                )
                if rewards
                else None
            ),
            "maximum": (
                max(rewards)
                if rewards
                else None
            ),
        },
    }

    with Path(
        manifest_path
    ).open(
        "w",
        encoding="utf-8",
    ) as output:

        json.dump(
            manifest,
            output,
            ensure_ascii=False,
            indent=2,
        )

        output.write("\n")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build state-action value "
            "training records for the "
            "Kaggriculture economic policy."
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
    )

    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
    )

    args = parser.parse_args()

    manifest = build_value_dataset(
        input_path=args.input,
        output_path=args.output,
        schema_path=args.schema,
        manifest_path=args.manifest,
    )

    print(
        "Economic value dataset built:"
    )

    print(
        f"  records:  "
        f"{manifest['records']}"
    )

    print(
        f"  episodes: "
        f"{manifest['episodes']}"
    )

    print(
        f"  output:   "
        f"{args.output}"
    )

    margin = manifest[
        "margin"
    ]

    print(
        "  margin:   "
        f"min={margin['minimum']:.1f}, "
        f"mean={margin['mean']:.1f}, "
        f"max={margin['maximum']:.1f}"
    )


if __name__ == "__main__":
    main()