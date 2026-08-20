"""Convert Kaggriculture observations into stable numeric feature vectors.

The vector is sparse: only non-zero indices and their values are stored.  Its
layout is described by ``feature_schema.json``, so training code can restore a
dense vector or split it into scalar and spatial inputs without guessing.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from src.kaggriculture.core.game_data import ANIMAL_SPECS, BASE_PRICES, CROP_SPECS, PRODUCTS, SHOP_PRODUCTS
from src.kaggriculture.core.state_parser import (
    COOP,
    EMPTY,
    LOCKED,
    PASTURE,
    PLANT,
    WEED,
    DEFAULT_EPISODE_STEPS,
    FarmState,
    GameState,
    TileState,
    parse_observation,
)


QUADRANTS = ("NW", "NE", "SW", "SE")
CROPS = tuple(CROP_SPECS)
ANIMALS = tuple(ANIMAL_SPECS)
SHOPS = tuple(SHOP_PRODUCTS)
INVENTORY_ITEMS = (*PRODUCTS, *ANIMALS)

SCALAR_FEATURES = (
    "step_fraction",
    "day_fraction",
    "hour_fraction",
    "hour_sin",
    "hour_cos",
    "remaining_fraction",
    "me_money_log",
    "opponent_money_log",
    "money_margin_signed_log",
    "me_hand_count_log",
    "opponent_hand_count_log",
    "me_hires_today_log",
    "opponent_hires_today_log",
    "shed_fill_fraction",
    *(f"me_unlocked_{quadrant.lower()}" for quadrant in QUADRANTS),
    *(f"opponent_unlocked_{quadrant.lower()}" for quadrant in QUADRANTS),
    *(f"seed_{crop.lower()}_log" for crop in CROPS),
    *(f"shed_{item.lower()}_log" for item in INVENTORY_ITEMS),
    *(f"carried_{item.lower()}_log" for item in INVENTORY_ITEMS),
    *(f"market_inventory_{item.lower()}_log" for item in PRODUCTS),
    *(f"market_price_ratio_{item.lower()}" for item in PRODUCTS),
    *(f"shop_{shop.lower()}_count" for shop in SHOPS),
    *(f"town_demand_{item.lower()}" for item in PRODUCTS),
    *(
        f"{side}_{metric}"
        for side in ("me", "opponent")
        for metric in (
            "plant_count_log",
            "animal_count_log",
            "weed_count_log",
            "empty_count_log",
            "needs_water_count_log",
            "needs_feed_count_log",
            "yield_units_log",
        )
    ),
)

SPATIAL_CHANNELS = (
    "available",
    "empty",
    "locked",
    "weed",
    "plant",
    *(f"crop_{crop.lower()}" for crop in CROPS),
    "plant_age_fraction",
    "yield_units_scaled",
    "lifespan_remaining_fraction",
    "watered_today",
    "unwatered_streak_scaled",
    "fertilized_active",
    "coop",
    "pasture",
    "animal",
    *(f"animal_{animal.lower()}" for animal in ANIMALS),
    "fed_today",
    "unfed_streak_scaled",
    "cared_today",
    "fertilizer_available",
    "pending_care_bonus_scaled",
    "farmer",
    "hand_count",
)


@dataclass(frozen=True)
class SparseFeatures:
    size: int
    indices: tuple[int, ...]
    values: tuple[float, ...]

    def to_dense(self) -> list[float]:
        dense = [0.0] * self.size
        for index, value in zip(self.indices, self.values):
            dense[index] = value
        return dense


class FeatureExtractor:
    """Create one fixed-layout vector from an observation."""

    def __init__(self, board_size: int = 10):
        if board_size <= 0:
            raise ValueError("board_size must be positive")
        self.board_size = board_size
        self.scalar_index = {name: index for index, name in enumerate(SCALAR_FEATURES)}
        self.channel_index = {
            name: index for index, name in enumerate(SPATIAL_CHANNELS)
        }
        self.scalar_count = len(SCALAR_FEATURES)
        self.spatial_count = 2 * len(SPATIAL_CHANNELS) * board_size * board_size
        self.feature_count = self.scalar_count + self.spatial_count

    def schema(self) -> dict[str, Any]:
        return {
            "version": 1,
            "representation": "sparse_indices_and_values",
            "feature_count": self.feature_count,
            "scalar_count": self.scalar_count,
            "scalar_features": list(SCALAR_FEATURES),
            "spatial": {
                "farm_order": ["me", "opponent"],
                "channels": list(SPATIAL_CHANNELS),
                "shape": [2, len(SPATIAL_CHANNELS), self.board_size, self.board_size],
                "flattening": "farm, channel, y, x",
                "offset": self.scalar_count,
            },
            "normalization": {
                "money_log": "signed log1p divided by log(100001)",
                "item_count_log": "log1p divided by log(101)",
                "market_inventory_log": "log1p divided by log(10001)",
                "price_ratio": "current price divided by documented base price",
            },
        }

    def feature_name(self, index: int) -> str:
        if not 0 <= index < self.feature_count:
            raise IndexError(index)
        if index < self.scalar_count:
            return SCALAR_FEATURES[index]
        relative = index - self.scalar_count
        cells = self.board_size * self.board_size
        farm_index, relative = divmod(relative, len(SPATIAL_CHANNELS) * cells)
        channel_index, cell_index = divmod(relative, cells)
        y, x = divmod(cell_index, self.board_size)
        return (
            f"{'me' if farm_index == 0 else 'opponent'}."
            f"{SPATIAL_CHANNELS[channel_index]}[{y},{x}]"
        )

    def extract(
        self, observation: Any, configuration: Any | None = None
    ) -> SparseFeatures:
        state = (
            observation
            if isinstance(observation, GameState)
            else parse_observation(observation, configuration)
        )
        self._validate_board(state)
        values: dict[int, float] = {}

        def scalar(name: str, value: float) -> None:
            self._put(values, self.scalar_index[name], value)

        episode_denominator = max(1, state.episode_steps - 1)
        day_count = max(1, math.ceil(state.episode_steps / state.turns_per_day))
        hour_angle = 2 * math.pi * state.hour / max(1, state.turns_per_day)
        scalar("step_fraction", state.step / episode_denominator)
        scalar("day_fraction", state.day / max(1, day_count - 1))
        scalar("hour_fraction", state.hour / max(1, state.turns_per_day - 1))
        scalar("hour_sin", math.sin(hour_angle))
        scalar("hour_cos", math.cos(hour_angle))
        scalar(
            "remaining_fraction",
            max(0, state.episode_steps - 1 - state.step) / episode_denominator,
        )
        scalar("me_money_log", _money_scale(state.me.money))
        scalar("opponent_money_log", _money_scale(state.opponent.money))
        scalar(
            "money_margin_signed_log",
            _signed_money_scale(state.me.money - state.opponent.money),
        )
        scalar("me_hand_count_log", _count_scale(len(state.me.hands)))
        scalar("opponent_hand_count_log", _count_scale(len(state.opponent.hands)))
        scalar("me_hires_today_log", _count_scale(state.me.hires_today))
        scalar("opponent_hires_today_log", _count_scale(state.opponent.hires_today))
        scalar(
            "shed_fill_fraction",
            state.private.shed_used / max(1, state.private.shed_capacity),
        )

        for side, farm in (("me", state.me), ("opponent", state.opponent)):
            unlocked = set(farm.unlocked_quadrants)
            for quadrant in QUADRANTS:
                scalar(f"{side}_unlocked_{quadrant.lower()}", float(quadrant in unlocked))

        carried = _sum_inventories(state.private.inventories)
        for crop in CROPS:
            scalar(f"seed_{crop.lower()}_log", _count_scale(state.private.seeds.get(crop, 0)))
        for item in INVENTORY_ITEMS:
            scalar(f"shed_{item.lower()}_log", _count_scale(state.private.shed.get(item, 0)))
            scalar(f"carried_{item.lower()}_log", _count_scale(carried[item]))
        for item in PRODUCTS:
            scalar(
                f"market_inventory_{item.lower()}_log",
                _market_scale(state.market.inventory.get(item, 0)),
            )
            scalar(
                f"market_price_ratio_{item.lower()}",
                state.market.prices.get(item, BASE_PRICES[item]) / BASE_PRICES[item],
            )

        shop_counts = state.town.shop_counts
        demand_counts: Counter[str] = Counter()
        for shop, count in shop_counts.items():
            multiplier = 2 if len(SHOP_PRODUCTS.get(shop, ())) == 1 else 1
            for product in SHOP_PRODUCTS.get(shop, ()):
                demand_counts[product] += count * multiplier
        for shop in SHOPS:
            scalar(f"shop_{shop.lower()}_count", shop_counts.get(shop, 0) / 8)
        for item in PRODUCTS:
            scalar(f"town_demand_{item.lower()}", demand_counts[item] / 16)

        for side, farm in (("me", state.me), ("opponent", state.opponent)):
            metrics = _farm_metrics(farm)
            for metric, value in metrics.items():
                scalar(f"{side}_{metric}_log", _count_scale(value))

        self._encode_farm(values, state, state.me, farm_index=0)
        self._encode_farm(values, state, state.opponent, farm_index=1)
        ordered = sorted(values.items())
        return SparseFeatures(
            size=self.feature_count,
            indices=tuple(index for index, _ in ordered),
            values=tuple(value for _, value in ordered),
        )

    def _validate_board(self, state: GameState) -> None:
        for label, farm in (("me", state.me), ("opponent", state.opponent)):
            if farm.width != self.board_size or farm.height != self.board_size:
                raise ValueError(
                    f"{label} board is {farm.width}x{farm.height}; "
                    f"extractor expects {self.board_size}x{self.board_size}"
                )

    def _encode_farm(
        self,
        values: dict[int, float],
        state: GameState,
        farm: FarmState,
        farm_index: int,
    ) -> None:
        for tile in farm.all_tiles:
            if tile.kind != LOCKED:
                self._put_spatial(values, farm_index, "available", tile, 1)
            if tile.kind == EMPTY:
                self._put_spatial(values, farm_index, "empty", tile, 1)
            elif tile.kind == LOCKED:
                self._put_spatial(values, farm_index, "locked", tile, 1)
            elif tile.kind == WEED:
                self._put_spatial(values, farm_index, "weed", tile, 1)
            elif tile.kind == PLANT:
                self._encode_plant(values, state, tile, farm_index)
            elif tile.kind in {COOP, PASTURE}:
                self._encode_structure(values, tile, farm_index)

        self._put_position(values, farm_index, "farmer", farm.farmer.x, farm.farmer.y, 1)
        hand_positions = Counter((hand.x, hand.y) for hand in farm.hands)
        for (x, y), count in hand_positions.items():
            self._put_position(values, farm_index, "hand_count", x, y, count)

    def _encode_plant(
        self,
        values: dict[int, float],
        state: GameState,
        tile: TileState,
        farm_index: int,
    ) -> None:
        self._put_spatial(values, farm_index, "plant", tile, 1)
        if tile.crop in CROP_SPECS:
            self._put_spatial(values, farm_index, f"crop_{tile.crop.lower()}", tile, 1)
        if tile.planted_day is not None:
            self._put_spatial(
                values,
                farm_index,
                "plant_age_fraction",
                tile,
                max(0, state.day - tile.planted_day) / 30,
            )
        self._put_spatial(values, farm_index, "yield_units_scaled", tile, tile.yield_units / 10)
        if tile.max_lifespan_step >= 0:
            self._put_spatial(
                values,
                farm_index,
                "lifespan_remaining_fraction",
                tile,
                (tile.max_lifespan_step - state.step) / max(1, state.episode_steps),
            )
        self._put_spatial(values, farm_index, "watered_today", tile, tile.watered_today)
        self._put_spatial(
            values,
            farm_index,
            "unwatered_streak_scaled",
            tile,
            tile.consecutive_unwatered / 2,
        )
        self._put_spatial(
            values,
            farm_index,
            "fertilized_active",
            tile,
            tile.fertilized_until_day >= state.day,
        )

    def _encode_structure(
        self, values: dict[int, float], tile: TileState, farm_index: int
    ) -> None:
        self._put_spatial(values, farm_index, tile.kind.lower(), tile, 1)
        if not tile.has_animal:
            return
        self._put_spatial(values, farm_index, "animal", tile, 1)
        if tile.animal in ANIMAL_SPECS:
            self._put_spatial(
                values, farm_index, f"animal_{tile.animal.lower()}", tile, 1
            )
        self._put_spatial(values, farm_index, "yield_units_scaled", tile, tile.yield_units / 10)
        self._put_spatial(values, farm_index, "fed_today", tile, tile.fed_today)
        self._put_spatial(
            values,
            farm_index,
            "unfed_streak_scaled",
            tile,
            tile.consecutive_unfed / 2,
        )
        self._put_spatial(values, farm_index, "cared_today", tile, tile.cared_today)
        self._put_spatial(
            values,
            farm_index,
            "fertilizer_available",
            tile,
            tile.fertilizer_available,
        )
        self._put_spatial(
            values,
            farm_index,
            "pending_care_bonus_scaled",
            tile,
            tile.pending_care_bonus / 10,
        )

    def _put_spatial(
        self,
        values: dict[int, float],
        farm_index: int,
        channel: str,
        tile: TileState,
        value: float | bool,
    ) -> None:
        self._put_position(
            values, farm_index, channel, tile.position.x, tile.position.y, value
        )

    def _put_position(
        self,
        values: dict[int, float],
        farm_index: int,
        channel: str,
        x: int,
        y: int,
        value: float | bool,
    ) -> None:
        cells = self.board_size * self.board_size
        index = (
            self.scalar_count
            + farm_index * len(SPATIAL_CHANNELS) * cells
            + self.channel_index[channel] * cells
            + y * self.board_size
            + x
        )
        self._put(values, index, value)

    @staticmethod
    def _put(values: dict[int, float], index: int, value: float | bool) -> None:
        number = round(float(value), 6)
        if not math.isfinite(number):
            raise ValueError(f"Feature {index} is not finite: {number}")
        if number != 0:
            values[index] = number


def _sum_inventories(inventories: Iterable[Mapping[str, int]]) -> Counter[str]:
    total: Counter[str] = Counter()
    for inventory in inventories:
        total.update(inventory)
    return total


def _farm_metrics(farm: FarmState) -> dict[str, int]:
    return {
        "plant_count": len(farm.plants),
        "animal_count": len(farm.animals),
        "weed_count": len(farm.weeds),
        "empty_count": len(farm.empty_tiles),
        "needs_water_count": sum(tile.needs_water for tile in farm.plants),
        "needs_feed_count": sum(tile.needs_feed for tile in farm.animals),
        "yield_units": sum(tile.yield_units for tile in (*farm.plants, *farm.animals)),
    }


def _money_scale(value: float) -> float:
    return math.log1p(max(0, value)) / math.log(100001)


def _signed_money_scale(value: float) -> float:
    return math.copysign(math.log1p(abs(value)) / math.log(100001), value)


def _count_scale(value: int) -> float:
    return math.log1p(max(0, value)) / math.log(101)


def _market_scale(value: int) -> float:
    return math.log1p(max(0, value)) / math.log(10001)


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", compresslevel=1)
    return path.open(mode, encoding="utf-8")


def build_feature_dataset(
    transitions_path: str | Path,
    output_path: str | Path,
    schema_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    board_size: int = 10,
) -> dict[str, Any]:
    """Transform replay transition records into sparse numeric records."""

    transitions_path = Path(transitions_path)
    output_path = Path(output_path)
    schema_path = Path(schema_path) if schema_path else output_path.with_name("feature_schema.json")
    manifest_path = (
        Path(manifest_path) if manifest_path else output_path.with_name("feature_manifest.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extractor = FeatureExtractor(board_size=board_size)
    record_count = 0
    nonzero_counts: list[int] = []
    cached_key: tuple[int, int, int] | None = None
    cached_features: SparseFeatures | None = None

    with _open_text(transitions_path, "rt") as source, _open_text(output_path, "wt") as output:
        for line_number, line in enumerate(source, start=1):
            try:
                transition = json.loads(line)
                current_key = (
                    int(transition["episode_id"]),
                    int(transition["seat"]),
                    int(transition["step"]),
                )
                features = (
                    cached_features
                    if cached_key == current_key and cached_features is not None
                    else extractor.extract(transition["observation"])
                )
                next_features = extractor.extract(transition["next_observation"])
                cached_key = (
                    current_key[0],
                    current_key[1],
                    int(transition["next_observation"]["step"]),
                )
                cached_features = next_features
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid transition on line {line_number}: {exc}") from exc
            record = {
                "episode_id": transition["episode_id"],
                "episode_type": transition["episode_type"],
                "seat": transition["seat"],
                "step": transition["step"],
                "feature_indices": features.indices,
                "feature_values": features.values,
                "action": transition["action"],
                "next_feature_indices": next_features.indices,
                "next_feature_values": next_features.values,
                "terminal": (
                    transition["next_observation"]["step"]
                    >= DEFAULT_EPISODE_STEPS - 1
                ),
                "final_reward": transition["final_reward"],
                "margin": transition["margin"],
                "outcome": transition["outcome"],
            }
            json.dump(record, output, separators=(",", ":"))
            output.write("\n")
            record_count += 1
            nonzero_counts.extend((len(features.indices), len(next_features.indices)))
            if record_count % 10_000 == 0:
                print(f"[features] {record_count:,} transitions processed", flush=True)

    schema = extractor.schema()
    with schema_path.open("w", encoding="utf-8") as output:
        json.dump(schema, output, ensure_ascii=True, indent=2)
        output.write("\n")

    manifest = {
        "source": str(transitions_path.resolve()),
        "output": output_path.name,
        "schema": schema_path.name,
        "records": record_count,
        "feature_count": extractor.feature_count,
        "average_nonzero_features": (
            round(sum(nonzero_counts) / len(nonzero_counts), 2) if nonzero_counts else 0
        ),
        "contains_next_features": True,
        "contains_terminal_flag": True,
        "action_encoding": "raw_kaggriculture_action",
    }
    with manifest_path.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=True, indent=2)
        output.write("\n")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "transitions", nargs="?", default="data/processed/transitions.jsonl.gz"
    )
    parser.add_argument(
        "--output", default="data/processed/features.jsonl.gz", dest="output"
    )
    parser.add_argument("--schema", default=None)
    parser.add_argument("--board-size", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_feature_dataset(
        args.transitions, args.output, schema_path=args.schema, board_size=args.board_size
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
