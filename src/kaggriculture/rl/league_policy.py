"""Opponent-aware market residual policy for the Stage 3 league agent."""

from __future__ import annotations

import copy
import math
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..core.game_data import (
    ANIMAL_SPECS,
    BASE_PRICES,
    CROP_SPECS,
    PRODUCTS,
    SHOP_PRODUCTS,
)
from .meta_policy import MetaControllerAgent, call_agent, normalize_action


LEAGUE_POLICY_VERSION = 5
RESIDUAL_PRODUCTS = ("MILK", "WOOL", "STRAWBERRY", "MELON")
ACTION_NAMES = ("KEEP_INCUMBENT", "HOLD", "SELL_25", "SELL_50", "SELL_100")
TARGET_SALE_FRACTIONS = (None, 0.0, 0.25, 0.5, 1.0)
ACTION_DIMS = (5, 5, 5, 5)
ENDGAME_LIQUIDATION_DAY = 27
HISTORY_LAGS = (1, 4, 24)
FARM_DELTA_LAGS = (1, 24)
FARM_METRICS = (
    "money",
    "hands",
    "land",
    "animals",
    "plants",
    "premium_plants",
    "weeds",
    "yield_units",
)


def _feature_names() -> tuple[str, ...]:
    names = [
        "step_fraction",
        "day_fraction",
        "hour_fraction",
        "hour_sin",
        "hour_cos",
        "remaining_fraction",
        "me_money",
        "opponent_money",
        "money_margin",
        "me_hands",
        "opponent_hands",
        "me_land",
        "opponent_land",
    ]
    for side in ("me", "opponent"):
        for animal in ANIMAL_SPECS:
            names.append(f"{side}_animal_{animal.lower()}")
        for crop in CROP_SPECS:
            names.append(f"{side}_crop_{crop.lower()}")
        names.extend(
            [
                f"{side}_weeds",
                f"{side}_yield_units",
            ]
        )
    for product in PRODUCTS:
        low = product.lower()
        names.extend(
            [
                f"shed_{low}",
                f"carried_{low}",
                f"market_inventory_{low}",
                f"market_price_{low}",
                f"town_demand_{low}",
            ]
        )
    for lag in HISTORY_LAGS:
        for product in PRODUCTS:
            low = product.lower()
            names.extend(
                [
                    f"inventory_delta_{lag}_{low}",
                    f"price_delta_{lag}_{low}",
                ]
            )
    for lag in FARM_DELTA_LAGS:
        for metric in FARM_METRICS:
            names.append(f"opponent_delta_{lag}_{metric}")
    for product in RESIDUAL_PRODUCTS:
        names.extend(
            [
                f"available_{product.lower()}",
                f"incumbent_sell_{product.lower()}",
            ]
        )
        for choice in range(len(ACTION_NAMES)):
            names.append(f"previous_{product.lower()}_{choice}")
    return tuple(names)


FEATURE_NAMES = _feature_names()


def _signed_log(value: float, divisor: float = 10.0) -> float:
    return float(np.clip(math.copysign(math.log1p(abs(value)), value) / divisor, -10, 10))


def _farm_snapshot(farm: Mapping[str, Any]) -> dict[str, Any]:
    animals: Counter[str] = Counter()
    crops: Counter[str] = Counter()
    weeds = yield_units = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if not isinstance(tile, Mapping):
                continue
            if tile.get("animal"):
                animals[str(tile["animal"])] += 1
            if tile.get("kind") == "PLANT" and tile.get("crop"):
                crops[str(tile["crop"])] += 1
            weeds += tile.get("kind") == "WEED"
            yield_units += max(0, int(tile.get("yield_units", 0) or 0))
    return {
        "money": float(farm.get("money", 0) or 0),
        "hands": len(farm.get("hands", []) or []),
        "land": len(farm.get("unlocked_quadrants", []) or []),
        "animals": sum(animals.values()),
        "plants": sum(crops.values()),
        "premium_plants": crops["STRAWBERRY"] + crops["MELON"],
        "weeds": weeds,
        "yield_units": yield_units,
        "animal_counts": animals,
        "crop_counts": crops,
    }


def _observation_snapshot(observation: Mapping[str, Any]) -> dict[str, Any]:
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms", []) or []
    me = farms[player]
    opponent = farms[1 - player]
    private = observation.get("private", {}) or {}
    carried: Counter[str] = Counter()
    for inventory in private.get("inventories", []) or []:
        if isinstance(inventory, Mapping):
            carried.update({str(key): int(value or 0) for key, value in inventory.items()})
    market = observation.get("market", {}) or {}
    shops = observation.get("town", {}).get("unlocked_shops", []) or []
    demand: Counter[str] = Counter()
    for shop in shops:
        products = SHOP_PRODUCTS.get(str(shop), ())
        multiplier = 2 if len(products) == 1 else 1
        for product in products:
            demand[product] += multiplier
    step = int(
        observation.get(
            "step",
            int(observation.get("day", 0) or 0) * 24
            + int(observation.get("hour", 0) or 0),
        )
        or 0
    )
    return {
        "step": step,
        "day": int(observation.get("day", step // 24) or 0),
        "hour": int(observation.get("hour", step % 24) or 0),
        "me": _farm_snapshot(me),
        "opponent": _farm_snapshot(opponent),
        "shed": {str(key): int(value or 0) for key, value in private.get("shed", {}).items()},
        "carried": dict(carried),
        "inventory": {
            product: int(market.get("inventory", {}).get(product, 0) or 0)
            for product in PRODUCTS
        },
        "prices": {
            product: int(market.get("prices", {}).get(product, 0) or 0)
            for product in PRODUCTS
        },
        "demand": dict(demand),
    }


class MarketHistoryFeatures:
    """Compact state plus public-market and opponent history."""

    feature_names = FEATURE_NAMES
    feature_count = len(FEATURE_NAMES)

    def __init__(self) -> None:
        self.history: deque[dict[str, Any]] = deque(maxlen=max(HISTORY_LAGS) + 1)
        self.previous_choices = (0,) * len(RESIDUAL_PRODUCTS)

    def reset(self) -> None:
        self.history.clear()
        self.previous_choices = (0,) * len(RESIDUAL_PRODUCTS)

    def record_action(self, choices: Sequence[int]) -> None:
        if len(choices) != len(RESIDUAL_PRODUCTS):
            raise ValueError("Wrong residual choice count")
        self.previous_choices = tuple(int(choice) for choice in choices)

    def _lagged(self, step: int, lag: int) -> dict[str, Any]:
        target = step - lag
        for snapshot in reversed(self.history):
            if int(snapshot["step"]) <= target:
                return snapshot
        return self.history[0]

    def extract(
        self,
        observation: Mapping[str, Any],
        base_action: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        current = _observation_snapshot(observation)
        if not self.history or self.history[-1]["step"] != current["step"]:
            self.history.append(current)
        else:
            self.history[-1] = current
        step = current["step"]
        hour_angle = 2.0 * math.pi * current["hour"] / 24.0
        me = current["me"]
        opponent = current["opponent"]
        values = [
            step / 720.0,
            current["day"] / 30.0,
            current["hour"] / 24.0,
            math.sin(hour_angle),
            math.cos(hour_angle),
            max(0.0, 720.0 - step) / 720.0,
            _signed_log(me["money"]),
            _signed_log(opponent["money"]),
            _signed_log(me["money"] - opponent["money"]),
            me["hands"] / 12.0,
            opponent["hands"] / 12.0,
            me["land"] / 4.0,
            opponent["land"] / 4.0,
        ]
        for farm in (me, opponent):
            values.extend(farm["animal_counts"][animal] / 16.0 for animal in ANIMAL_SPECS)
            values.extend(farm["crop_counts"][crop] / 25.0 for crop in CROP_SPECS)
            values.extend([farm["weeds"] / 25.0, _signed_log(farm["yield_units"], 6.0)])
        for product in PRODUCTS:
            values.extend(
                [
                    current["shed"].get(product, 0) / 100.0,
                    current["carried"].get(product, 0) / 100.0,
                    _signed_log(current["inventory"][product], 8.0),
                    current["prices"][product] / max(1.0, BASE_PRICES[product]),
                    current["demand"].get(product, 0) / 8.0,
                ]
            )
        for lag in HISTORY_LAGS:
            previous = self._lagged(step, lag)
            for product in PRODUCTS:
                values.extend(
                    [
                        _signed_log(
                            current["inventory"][product] - previous["inventory"][product],
                            6.0,
                        ),
                        (current["prices"][product] - previous["prices"][product])
                        / max(1.0, BASE_PRICES[product]),
                    ]
                )
        for lag in FARM_DELTA_LAGS:
            previous = self._lagged(step, lag)["opponent"]
            for metric in FARM_METRICS:
                difference = opponent[metric] - previous[metric]
                values.append(
                    _signed_log(difference, 8.0)
                    if metric == "money"
                    else float(difference) / 25.0
                )
        player = int(observation.get("player", 0) or 0)
        hand_count = len(observation["farms"][player].get("hands", []) or [])
        normalized = normalize_action(base_action or {}, hand_count)
        reservations = _pickup_reservations(normalized)
        incumbent_sales = premium_sell_quantities(normalized)
        for product in RESIDUAL_PRODUCTS:
            available = max(
                0,
                int(current["shed"].get(product, 0)) - reservations.get(product, 0),
            )
            values.extend([available / 100.0, incumbent_sales[product] / 100.0])
        for product_index in range(len(RESIDUAL_PRODUCTS)):
            selected = self.previous_choices[product_index]
            values.extend(
                float(choice == selected) for choice in range(len(ACTION_NAMES))
            )
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (self.feature_count,):
            raise RuntimeError(
                f"History feature layout mismatch: {result.shape} != {(self.feature_count,)}"
            )
        return np.clip(result, -10.0, 10.0)


def _pickup_reservations(action: Mapping[str, Any]) -> Counter[str]:
    reservations: Counter[str] = Counter()
    for command in [action.get("farmer", ["PASS"]), *(action.get("hands", []) or [])]:
        if not isinstance(command, (list, tuple)) or len(command) < 2 or command[0] != "PICKUP":
            continue
        reservations[str(command[1])] += int(command[2]) if len(command) > 2 else 1
    return reservations


def premium_sell_quantities(action: Mapping[str, Any]) -> Counter[str]:
    """Return total premium-product quantities in market SELL orders."""

    quantities: Counter[str] = Counter()
    for order in action.get("market", []) or []:
        if (
            isinstance(order, (list, tuple))
            and len(order) >= 3
            and order[0] == "SELL"
            and order[1] in RESIDUAL_PRODUCTS
        ):
            quantities[str(order[1])] += max(0, int(order[2]))
    return quantities


def market_decision_available(
    base_action: Mapping[str, Any], observation: Mapping[str, Any]
) -> bool:
    """Whether at least one policy choice can change this turn's orders."""

    player = int(observation.get("player", 0) or 0)
    hand_count = len(observation["farms"][player].get("hands", []) or [])
    normalized = normalize_action(base_action, hand_count)
    incumbent_sales = premium_sell_quantities(normalized)
    if any(incumbent_sales.values()):
        return True
    reservations = _pickup_reservations(normalized)
    shed = observation.get("private", {}).get("shed", {}) or {}
    room_for_order = len(normalized["market"]) < 10
    return room_for_order and any(
        int(shed.get(product, 0) or 0) - reservations.get(product, 0) > 0
        for product in RESIDUAL_PRODUCTS
    )


def enforce_endgame_liquidation(
    observation: Mapping[str, Any], choices: Sequence[int]
) -> tuple[int, ...]:
    """Turn every active residual into full liquidation near season end.

    Choice zero remains an exact incumbent fallback. A learned policy therefore
    cannot hold stock forever, while the safe fallback remains byte-for-byte
    equivalent in action semantics.
    """

    normalized = tuple(int(choice) for choice in choices)
    if len(normalized) != len(RESIDUAL_PRODUCTS):
        raise ValueError("Wrong residual choice count")
    day = int(observation.get("day", 0) or 0)
    if day < ENDGAME_LIQUIDATION_DAY:
        return normalized
    sell_all = len(ACTION_NAMES) - 1
    return tuple(0 if choice == 0 else sell_all for choice in normalized)


def apply_market_residual(
    base_action: Mapping[str, Any],
    observation: Mapping[str, Any],
    choices: Sequence[int],
) -> dict[str, list[Any]]:
    """Override premium sale quantities while preserving every non-sale command."""

    choices = enforce_endgame_liquidation(observation, choices)
    player = int(observation.get("player", 0) or 0)
    hand_count = len(observation["farms"][player].get("hands", []) or [])
    action = normalize_action(copy.deepcopy(base_action), hand_count)
    market = action["market"]
    shed = observation.get("private", {}).get("shed", {}) or {}
    reserved = _pickup_reservations(action)
    for product, raw_choice in zip(RESIDUAL_PRODUCTS, choices):
        choice = int(raw_choice)
        if not 0 <= choice < len(TARGET_SALE_FRACTIONS):
            raise ValueError(f"Invalid residual choice {choice}")
        fraction = TARGET_SALE_FRACTIONS[choice]
        if fraction is None:
            continue
        available = max(
            0,
            int(shed.get(product, 0) or 0)
            - reserved.get(product, 0),
        )
        quantity = int(math.ceil(available * fraction)) if available else 0
        matching_indices = [
            index
            for index, order in enumerate(market)
            if len(order) >= 3 and order[0] == "SELL" and order[1] == product
        ]
        if matching_indices:
            first = matching_indices[0]
            if quantity > 0:
                market[first] = ["SELL", product, quantity]
                market = [
                    order
                    for index, order in enumerate(market)
                    if index == first or index not in matching_indices
                ]
            else:
                market = [
                    order
                    for index, order in enumerate(market)
                    if index not in matching_indices
                ]
        elif quantity > 0 and len(market) < 10:
            market.append(["SELL", product, quantity])
    action["market"] = market[:10]
    return action


class NumpyLeaguePolicy:
    """Pure NumPy inference for an SB3 MultiDiscrete actor."""

    def __init__(self, model_path: str | Path) -> None:
        with np.load(model_path) as payload:
            self.weights = tuple(
                (
                    np.asarray(payload[f"w{index}"], dtype=np.float32),
                    np.asarray(payload[f"b{index}"], dtype=np.float32),
                )
                for index in range(3)
            )
            self.feature_count = int(payload["feature_count"])
            self.action_dims = tuple(int(value) for value in payload["action_dims"])
            self.policy_version = int(payload.get("policy_version", 0))
        if self.policy_version != LEAGUE_POLICY_VERSION:
            raise ValueError(
                f"League policy version {self.policy_version}; expected {LEAGUE_POLICY_VERSION}"
            )
        if self.feature_count != len(FEATURE_NAMES):
            raise ValueError("League feature schema does not match the model")
        if self.action_dims != ACTION_DIMS:
            raise ValueError("League action schema does not match the model")

    def logits(self, features: Sequence[float]) -> np.ndarray:
        value = np.asarray(features, dtype=np.float32)
        if value.shape != (self.feature_count,):
            raise ValueError(f"Expected {self.feature_count} features, got {value.shape}")
        for index, (weight, bias) in enumerate(self.weights):
            value = weight @ value + bias
            if index < len(self.weights) - 1:
                value = np.tanh(value)
        return value

    def predict(self, features: Sequence[float]) -> np.ndarray:
        logits = self.logits(features)
        choices = []
        offset = 0
        for size in self.action_dims:
            choices.append(int(np.argmax(logits[offset : offset + size])))
            offset += size
        return np.asarray(choices, dtype=np.int64)


class LeagueResidualAgent:
    """Incumbent v2 route plus an opponent-aware market-only residual."""

    def __init__(
        self,
        league_model_path: str | Path,
        incumbent_model_path: str | Path,
        expert_path: str | Path,
    ) -> None:
        self.policy = NumpyLeaguePolicy(league_model_path)
        self.incumbent = MetaControllerAgent(incumbent_model_path, expert_path)
        self.features = MarketHistoryFeatures()

    def __call__(
        self, observation: Mapping[str, Any], configuration: Any | None = None
    ) -> dict[str, Any]:
        step = int(
            observation.get(
                "step",
                int(observation.get("day", 0) or 0) * 24
                + int(observation.get("hour", 0) or 0),
            )
            or 0
        )
        if step == 0:
            self.features.reset()
        base = call_agent(self.incumbent, observation, configuration)
        vector = self.features.extract(observation, base)
        choices = enforce_endgame_liquidation(
            observation, self.policy.predict(vector)
        )
        action = apply_market_residual(base, observation, choices)
        self.features.record_action(choices)
        return action
