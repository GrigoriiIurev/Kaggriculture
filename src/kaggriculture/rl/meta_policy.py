"""Small NumPy policy that chooses between complete candidate actions."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import sys
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence

import numpy as np

from ..data.feature_extractor import FeatureExtractor


POLICY_VERSION = 2
PREMIUM_PRODUCTS = ("MILK", "WOOL", "STRAWBERRY", "MELON")
CANDIDATE_NAMES = (
    "expert",
    "expert_plus_25pct_premium_sales",
    "expert_plus_50pct_premium_sales",
    "expert_plus_100pct_premium_sales",
)


def call_agent(
    agent: Callable[..., dict[str, Any]],
    observation: Any,
    configuration: Any | None = None,
) -> dict[str, Any]:
    """Call either Kaggle's one-argument or two-argument agent convention."""

    parameters = inspect.signature(agent).parameters.values()
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        parameter.kind == parameter.VAR_POSITIONAL
        for parameter in parameters
    )
    if has_varargs or len(positional) >= 2:
        return agent(observation, configuration)
    return agent(observation)


def load_agent_module(path: str | Path, name: str | None = None) -> ModuleType:
    """Load an agent file under a unique name so its global state is isolated."""

    path = Path(path)
    module_name = name or f"kaggriculture_external_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load agent module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "agent", None)):
        raise AttributeError(f"{path} does not define a callable agent")
    return module


def normalize_action(action: Any, hand_count: int) -> dict[str, list[Any]]:
    """Return a detached, shape-safe Kaggriculture action."""

    action = action if isinstance(action, dict) else {}
    farmer = action.get("farmer", ["PASS"])
    farmer = list(farmer) if isinstance(farmer, (list, tuple)) else ["PASS"]
    raw_hands = action.get("hands", [])
    raw_hands = raw_hands if isinstance(raw_hands, (list, tuple)) else []
    hands = [
        list(raw_hands[index])
        if index < len(raw_hands) and isinstance(raw_hands[index], (list, tuple))
        else ["PASS"]
        for index in range(hand_count)
    ]
    market = action.get("market", [])
    market = market if isinstance(market, (list, tuple)) else []
    return {
        "farmer": farmer or ["PASS"],
        "hands": hands,
        "market": [list(order) for order in market if isinstance(order, (list, tuple))],
    }


def candidate_actions(
    observation: Any,
    expert: Callable[..., dict[str, Any]],
    configuration: Any | None = None,
) -> tuple[dict[str, list[Any]], ...]:
    """Keep the expert route intact and vary only extra premium sales."""

    player = int(observation["player"])
    hand_count = len(observation["farms"][player]["hands"])
    expert_action = normalize_action(
        call_agent(expert, observation, configuration), hand_count
    )
    return tuple(
        copy.deepcopy(expert_action)
        if fraction == 0.0
        else _add_premium_sales(expert_action, observation, fraction)
        for fraction in (0.0, 0.25, 0.5, 1.0)
    )


def _add_premium_sales(
    expert_action: dict[str, list[Any]],
    observation: Any,
    fraction: float,
) -> dict[str, list[Any]]:
    """Add legal sales without removing or reordering any expert command."""

    action = copy.deepcopy(expert_action)
    market = action["market"]
    shed = observation["private"]["shed"]
    reserved = _pickup_reservations(action)
    for product in PREMIUM_PRODUCTS:
        stock = max(0, int(shed.get(product, 0)))
        existing = next(
            (
                order
                for order in market
                if len(order) >= 3
                and order[0] == "SELL"
                and order[1] == product
            ),
            None,
        )
        already_selling = max(0, int(existing[2])) if existing else 0
        available = max(
            0,
            stock - already_selling - reserved.get(product, 0),
        )
        if available <= 0:
            continue
        quantity = max(1, int(np.ceil(available * fraction)))
        if existing is not None:
            existing[2] = already_selling + quantity
        elif len(market) < 10:
            market.append(["SELL", product, quantity])
    action["market"] = market[:10]
    return action


def _pickup_reservations(action: dict[str, list[Any]]) -> dict[str, int]:
    reservations: dict[str, int] = {}
    commands = [action["farmer"], *action["hands"]]
    for command in commands:
        if len(command) < 2 or command[0] != "PICKUP":
            continue
        item = str(command[1])
        quantity = 1 if len(command) < 3 else max(0, int(command[2]))
        reservations[item] = reservations.get(item, 0) + quantity
    return reservations


class NumpyMetaPolicy:
    """Two hidden-layer tanh policy exported from Stable-Baselines3 PPO."""

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
            self.candidate_count = int(payload["candidate_count"])
            self.policy_version = int(payload.get("policy_version", 0))
        if self.policy_version != POLICY_VERSION:
            raise ValueError(
                f"Model policy version is {self.policy_version}; "
                f"runtime requires {POLICY_VERSION}"
            )
        if self.candidate_count != len(CANDIDATE_NAMES):
            raise ValueError(
                f"Model has {self.candidate_count} candidates; "
                f"runtime has {len(CANDIDATE_NAMES)}"
            )

    def logits(self, features: Sequence[float]) -> np.ndarray:
        value = np.asarray(features, dtype=np.float32)
        if value.shape != (self.feature_count,):
            raise ValueError(
                f"Expected {self.feature_count} features, got {value.shape}"
            )
        for index, (weight, bias) in enumerate(self.weights):
            value = weight @ value + bias
            if index < len(self.weights) - 1:
                value = np.tanh(value)
        return value

    def predict(self, features: Sequence[float]) -> int:
        return int(np.argmax(self.logits(features)))


class MetaControllerAgent:
    """Submission agent: strong expert plus a learned candidate selector."""

    def __init__(self, model_path: str | Path, expert_path: str | Path) -> None:
        self.policy = NumpyMetaPolicy(model_path)
        self.extractor = FeatureExtractor()
        self.expert_module = load_agent_module(expert_path)

    def __call__(
        self, observation: Any, configuration: Any | None = None
    ) -> dict[str, Any]:
        features = self.extractor.extract(observation, configuration).to_dense()
        choice = self.policy.predict(features)
        actions = candidate_actions(
            observation,
            self.expert_module.agent,
            configuration,
        )
        return actions[choice]
