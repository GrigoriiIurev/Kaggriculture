"""Build per-worker behavior-cloning targets from replay transitions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from src.kaggriculture.core.action_codec import ActionEncoder, WORKER_OPERATIONS
from src.kaggriculture.data.feature_extractor import FeatureExtractor, INVENTORY_ITEMS, SparseFeatures
from src.kaggriculture.core.state_parser import GameState, parse_observation


WORKER_CONTEXT_FEATURES = (
    "is_farmer",
    "worker_index_fraction",
    "worker_count_fraction",
    "worker_x_fraction",
    "worker_y_fraction",
    *(f"worker_inventory_{item.lower()}_log" for item in INVENTORY_ITEMS),
)


class WorkerFeatureExtractor:
    """Append focused worker identity and inventory to the global state vector."""

    def __init__(self, board_size: int = 10):
        self.base = FeatureExtractor(board_size=board_size)
        self.board_size = board_size
        self.context_offset = self.base.feature_count
        self.focus_offset = self.context_offset + len(WORKER_CONTEXT_FEATURES)
        self.feature_count = self.focus_offset + board_size * board_size
        self.context_index = {
            name: self.context_offset + index
            for index, name in enumerate(WORKER_CONTEXT_FEATURES)
        }

    def schema(self) -> dict[str, Any]:
        return {
            "version": 1,
            "feature_count": self.feature_count,
            "base_feature_count": self.base.feature_count,
            "base_schema": self.base.schema(),
            "worker_context_offset": self.context_offset,
            "worker_context_features": list(WORKER_CONTEXT_FEATURES),
            "focus_position": {
                "offset": self.focus_offset,
                "shape": [self.board_size, self.board_size],
                "flattening": "y, x",
            },
        }

    def context(self, state: GameState, worker_index: int) -> SparseFeatures:
        if not 0 <= worker_index < len(state.units):
            raise IndexError(worker_index)
        worker = state.units[worker_index]
        values: dict[int, float] = {}
        self._put(values, "is_farmer", worker.is_farmer)
        self._put(values, "worker_index_fraction", worker.index / 20)
        self._put(values, "worker_count_fraction", len(state.units) / 20)
        self._put(
            values,
            "worker_x_fraction",
            worker.position.x / max(1, self.board_size - 1),
        )
        self._put(
            values,
            "worker_y_fraction",
            worker.position.y / max(1, self.board_size - 1),
        )
        for item in INVENTORY_ITEMS:
            self._put(
                values,
                f"worker_inventory_{item.lower()}_log",
                _count_scale(worker.inventory.get(item, 0)),
            )
        focus_index = self.focus_offset + worker.position.y * self.board_size + worker.position.x
        values[focus_index] = 1.0
        ordered = sorted(values.items())
        return SparseFeatures(
            size=self.feature_count,
            indices=tuple(index for index, _ in ordered),
            values=tuple(value for _, value in ordered),
        )

    def extract(self, observation: Any, worker_index: int) -> SparseFeatures:
        state = parse_observation(observation)
        base = self.base.extract(observation)
        context = self.context(state, worker_index)
        return SparseFeatures(
            size=self.feature_count,
            indices=(*base.indices, *context.indices),
            values=(*base.values, *context.values),
        )

    def feature_name(self, index: int) -> str:
        if index < self.base.feature_count:
            return self.base.feature_name(index)
        if index < self.focus_offset:
            return WORKER_CONTEXT_FEATURES[index - self.context_offset]
        if index < self.feature_count:
            y, x = divmod(index - self.focus_offset, self.board_size)
            return f"focused_worker[{y},{x}]"
        raise IndexError(index)

    def _put(self, values: dict[int, float], name: str, value: float | bool) -> None:
        number = round(float(value), 6)
        if not math.isfinite(number):
            raise ValueError(f"Worker feature {name} is not finite")
        if number != 0:
            values[self.context_index[name]] = number


def _count_scale(value: int) -> float:
    return math.log1p(max(0, value)) / math.log(101)


def episode_split(episode_id: int, holdout_fraction: float = 0.2) -> str:
    """Assign every transition from one episode to the same stable split."""

    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    digest = hashlib.sha256(str(episode_id).encode("ascii")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "holdout" if fraction < holdout_fraction else "train"


def build_worker_dataset(
    transitions_path: str | Path,
    output_path: str | Path,
    schema_path: str | Path | None = None,
    action_schema_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    board_size: int = 10,
) -> dict[str, Any]:
    transitions_path = Path(transitions_path)
    output_path = Path(output_path)
    schema_path = (
        Path(schema_path)
        if schema_path
        else output_path.with_name("worker_feature_schema.json")
    )
    action_schema_path = (
        Path(action_schema_path)
        if action_schema_path
        else output_path.with_name("action_schema.json")
    )
    manifest_path = (
        Path(manifest_path)
        if manifest_path
        else output_path.with_name("worker_manifest.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_extractor = WorkerFeatureExtractor(board_size=board_size)
    action_encoder = ActionEncoder()
    group_count = 0
    worker_count = 0
    split_groups: Counter[str] = Counter()
    split_workers: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()

    with _open_text(transitions_path, "rt") as source, _open_text(
        output_path, "wt"
    ) as output:
        for line_number, line in enumerate(source, start=1):
            try:
                transition = json.loads(line)
                observation = transition["observation"]
                state = parse_observation(observation)
                base_features = feature_extractor.base.extract(observation)
                encoded = action_encoder.encode_action(
                    transition["action"], expected_hands=len(state.me.hands)
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid transition on line {line_number}: {exc}") from exc

            commands = (encoded.farmer, *encoded.hands)
            raw_commands = (
                transition["action"]["farmer"],
                *transition["action"]["hands"],
            )
            workers = []
            for worker_index, (target, raw_target) in enumerate(
                zip(commands, raw_commands)
            ):
                context = feature_extractor.context(state, worker_index)
                workers.append(
                    {
                        "worker_index": worker_index,
                        "is_farmer": worker_index == 0,
                        "context_indices": context.indices,
                        "context_values": context.values,
                        "target": target.as_dict(),
                        "raw_target": raw_target,
                    }
                )
                operation_counts[WORKER_OPERATIONS[target.operation_id]] += 1

            split = episode_split(int(transition["episode_id"]))
            record = {
                "episode_id": transition["episode_id"],
                "episode_type": transition["episode_type"],
                "seat": transition["seat"],
                "step": transition["step"],
                "split": split,
                "feature_indices": base_features.indices,
                "feature_values": base_features.values,
                "workers": workers,
                "final_reward": transition["final_reward"],
                "margin": transition["margin"],
                "outcome": transition["outcome"],
            }
            json.dump(record, output, separators=(",", ":"))
            output.write("\n")
            group_count += 1
            worker_count += len(workers)
            split_groups[split] += 1
            split_workers[split] += len(workers)

    _write_json(schema_path, feature_extractor.schema())
    _write_json(action_schema_path, action_encoder.schema())
    manifest = {
        "source": str(transitions_path.resolve()),
        "output": output_path.name,
        "worker_feature_schema": schema_path.name,
        "action_schema": action_schema_path.name,
        "transition_groups": group_count,
        "worker_samples": worker_count,
        "feature_count": feature_extractor.feature_count,
        "split_groups": dict(split_groups),
        "split_worker_samples": dict(split_workers),
        "operation_counts": dict(operation_counts.most_common()),
        "future_outcome_is_not_an_input_feature": True,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", compresslevel=6)
    return path.open(mode, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=True, indent=2)
        output.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "transitions", nargs="?", default="data/processed/transitions.jsonl.gz"
    )
    parser.add_argument(
        "--output", default="data/processed/worker_dataset.jsonl.gz"
    )
    parser.add_argument("--board-size", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_worker_dataset(
        args.transitions, args.output, board_size=args.board_size
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
