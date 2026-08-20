"""Build economic-policy training targets from the global feature dataset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from src.kaggriculture.core.action_codec import (
    ARGUMENTS,
    MARKET_OPERATIONS,
    ActionEncoder,
)


DEFAULT_HOLDOUT_FRACTION = 0.2
DEFAULT_MAX_MARKET_ORDERS = 10


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", compresslevel=6)
    return path.open(mode, encoding="utf-8")


def episode_split(
    episode_id: int,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> str:
    """Assign every row from one episode to the same deterministic split."""

    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")

    digest = hashlib.sha256(str(episode_id).encode("ascii")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64

    return "holdout" if fraction < holdout_fraction else "train"


def _validate_sparse_features(
    indices: Any,
    values: Any,
    *,
    line_number: int,
) -> tuple[list[int], list[float]]:
    if not isinstance(indices, (list, tuple)):
        raise ValueError(
            f"feature_indices must be a list on line {line_number}"
        )

    if not isinstance(values, (list, tuple)):
        raise ValueError(
            f"feature_values must be a list on line {line_number}"
        )

    if len(indices) != len(values):
        raise ValueError(
            f"feature_indices and feature_values have different lengths "
            f"on line {line_number}"
        )

    clean_indices: list[int] = []
    clean_values: list[float] = []

    for index, value in zip(indices, values):
        index = int(index)
        value = float(value)

        if index < 0:
            raise ValueError(
                f"Negative feature index {index} on line {line_number}"
            )

        clean_indices.append(index)
        clean_values.append(value)

    return clean_indices, clean_values


def _encode_market_orders(
    raw_market: Any,
    encoder: ActionEncoder,
    *,
    line_number: int,
    max_market_orders: int,
) -> list[dict[str, Any]]:
    if raw_market is None:
        raw_market = []

    if not isinstance(raw_market, (list, tuple)):
        raise ValueError(
            f"action.market must be a list on line {line_number}"
        )

    if len(raw_market) > max_market_orders:
        raw_market = raw_market[:max_market_orders]

    encoded_orders: list[dict[str, Any]] = []

    for order_index, raw_order in enumerate(raw_market):
        encoded = encoder.encode_market(raw_order)

        encoded_orders.append(
            {
                "order_index": order_index,
                "operation_id": encoded.operation_id,
                "argument_id": encoded.argument_id,
                "quantity": encoded.quantity,
                "raw_order": list(raw_order),
            }
        )

    return encoded_orders


def _empty_market_target() -> dict[str, Any]:
    """
    Explicit target for a turn where the policy submitted no market orders.

    NO_ORDER is expected to be present in MARKET_OPERATIONS.
    """

    try:
        operation_id = MARKET_OPERATIONS.index("NO_ORDER")
    except ValueError as exc:
        raise ValueError(
            "MARKET_OPERATIONS must contain NO_ORDER"
        ) from exc

    return {
        "order_index": 0,
        "operation_id": operation_id,
        "argument_id": 0,
        "quantity": 0,
        "raw_order": [],
    }


def build_economic_dataset(
    features_path: str | Path,
    output_path: str | Path,
    schema_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    *,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    max_market_orders: int = DEFAULT_MAX_MARKET_ORDERS,
) -> dict[str, Any]:
    """
    Convert the global feature dataset into economic-policy examples.

    One output record corresponds to one game turn.

    Input:
        feature_indices / feature_values
        action.market
        episode metadata
        final_reward / margin / outcome

    Output:
        same state features
        encoded ordered market targets
        deterministic train/holdout split
        game outcome metadata

    Empty market turns receive one explicit NO_ORDER target.
    """

    features_path = Path(features_path)
    output_path = Path(output_path)

    if schema_path is None:
        schema_path = output_path.with_name("economic_action_schema.json")
    else:
        schema_path = Path(schema_path)

    if manifest_path is None:
        manifest_path = output_path.with_name("economic_manifest.json")
    else:
        manifest_path = Path(manifest_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = ActionEncoder()

    row_count = 0
    order_count = 0
    empty_market_count = 0

    split_rows: Counter[str] = Counter()
    split_orders: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    argument_counts: Counter[str] = Counter()
    quantity_counts: Counter[int] = Counter()
    outcome_counts: Counter[str] = Counter()
    orders_per_turn: Counter[int] = Counter()

    max_feature_index = -1
    episode_ids: set[int] = set()

    with _open_text(features_path, "rt") as source, _open_text(
        output_path, "wt"
    ) as output:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)

                episode_id = int(row["episode_id"])
                action = row["action"]

                if not isinstance(action, dict):
                    raise ValueError("action must be a mapping")

                feature_indices, feature_values = _validate_sparse_features(
                    row["feature_indices"],
                    row["feature_values"],
                    line_number=line_number,
                )

                raw_market = action.get("market", [])

                encoded_orders = _encode_market_orders(
                    raw_market,
                    encoder,
                    line_number=line_number,
                    max_market_orders=max_market_orders,
                )

            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                raise ValueError(
                    f"Invalid feature record on line {line_number}: {exc}"
                ) from exc

            split = episode_split(
                episode_id,
                holdout_fraction=holdout_fraction,
            )

            if encoded_orders:
                targets = encoded_orders
            else:
                targets = [_empty_market_target()]
                empty_market_count += 1

            record = {
                "episode_id": episode_id,
                "episode_type": row.get("episode_type"),
                "seat": row.get("seat"),
                "step": row.get("step"),
                "split": split,
                "feature_indices": feature_indices,
                "feature_values": feature_values,
                "targets": targets,
                "market_order_count": len(encoded_orders),
                "final_reward": row.get("final_reward"),
                "margin": row.get("margin"),
                "outcome": row.get("outcome"),
            }

            json.dump(
                record,
                output,
                ensure_ascii=True,
                separators=(",", ":"),
            )
            output.write("\n")

            row_count += 1
            episode_ids.add(episode_id)

            split_rows[split] += 1
            orders_per_turn[len(encoded_orders)] += 1

            outcome = row.get("outcome")
            if outcome is not None:
                outcome_counts[str(outcome)] += 1

            if feature_indices:
                max_feature_index = max(
                    max_feature_index,
                    max(feature_indices),
                )

            for target in targets:
                operation_id = int(target["operation_id"])

                if not 0 <= operation_id < len(MARKET_OPERATIONS):
                    raise ValueError(
                        f"Invalid market operation id {operation_id} "
                        f"on line {line_number}"
                    )

                operation_name = MARKET_OPERATIONS[operation_id]
                operation_counts[operation_name] += 1

                argument_id = int(target["argument_id"])

                if 0 <= argument_id < len(ARGUMENTS):
                    argument_name = ARGUMENTS[argument_id]
                else:
                    argument_name = f"UNKNOWN_{argument_id}"

                argument_counts[argument_name] += 1

                quantity = int(target["quantity"])
                quantity_counts[quantity] += 1

                order_count += 1
                split_orders[split] += 1

    schema = {
        "version": 1,
        "dataset": "economic_policy",
        "input": {
            "representation": "sparse",
            "feature_indices": "indices into feature_schema.json",
            "feature_values": "values at those indices",
            "observed_max_feature_index": max_feature_index,
        },
        "target": {
            "type": "ordered_market_commands",
            "max_market_orders": max_market_orders,
            "operations": list(MARKET_OPERATIONS),
            "arguments": list(ARGUMENTS),
            "fields": [
                "order_index",
                "operation_id",
                "argument_id",
                "quantity",
            ],
            "empty_turn": {
                "operation": "NO_ORDER",
                "quantity": 0,
            },
        },
        "split": {
            "method": "sha256_episode_id",
            "holdout_fraction": holdout_fraction,
        },
        "outcome_fields": [
            "final_reward",
            "margin",
            "outcome",
        ],
        "notes": [
            "Outcome fields are metadata and must not be used as inference features.",
            "All rows from one episode are assigned to the same split.",
            "Market command order is preserved.",
            "Turns without market commands receive an explicit NO_ORDER target.",
        ],
    }

    with Path(schema_path).open("w", encoding="utf-8") as output:
        json.dump(schema, output, ensure_ascii=True, indent=2)
        output.write("\n")

    manifest = {
        "version": 1,
        "source": str(features_path),
        "output": str(output_path),
        "episodes": len(episode_ids),
        "rows": row_count,
        "targets": order_count,
        "empty_market_turns": empty_market_count,
        "holdout_fraction": holdout_fraction,
        "split_rows": dict(sorted(split_rows.items())),
        "split_targets": dict(sorted(split_orders.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
        "argument_counts": dict(sorted(argument_counts.items())),
        "quantity_counts": {
            str(key): value
            for key, value in sorted(quantity_counts.items())
        },
        "orders_per_turn": {
            str(key): value
            for key, value in sorted(orders_per_turn.items())
        },
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "observed_max_feature_index": max_feature_index,
    }

    with Path(manifest_path).open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=True, indent=2)
        output.write("\n")

    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build economic-policy training targets from features.jsonl.gz"
        )
    )

    parser.add_argument(
        "features",
        nargs="?",
        default="data/processed/features.jsonl.gz",
        help="Input global feature dataset",
    )

    parser.add_argument(
        "--output",
        default="data/processed/economic_dataset.jsonl.gz",
        help="Output economic dataset",
    )

    parser.add_argument(
        "--schema",
        default=None,
        help="Optional output action-schema path",
    )

    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional output manifest path",
    )

    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=DEFAULT_HOLDOUT_FRACTION,
        help="Fraction of episodes assigned to holdout",
    )

    parser.add_argument(
        "--max-market-orders",
        type=int,
        default=DEFAULT_MAX_MARKET_ORDERS,
        help="Maximum number of market orders kept per turn",
    )

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.max_market_orders < 1:
        raise ValueError("--max-market-orders must be positive")

    manifest = build_economic_dataset(
        features_path=args.features,
        output_path=args.output,
        schema_path=args.schema,
        manifest_path=args.manifest,
        holdout_fraction=args.holdout_fraction,
        max_market_orders=args.max_market_orders,
    )

    print(
        "Economic dataset built:"
        f" {manifest['rows']} turns,"
        f" {manifest['targets']} targets,"
        f" {manifest['episodes']} episodes."
    )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()