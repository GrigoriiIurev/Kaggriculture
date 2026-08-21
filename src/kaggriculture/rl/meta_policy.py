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

from ..agent import RuleBasedAgent
from ..data.feature_extractor import FeatureExtractor


CANDIDATE_NAMES = (
    "expert",
    "rule_based",
    "expert_workers_rule_market",
    "rule_workers_expert_market",
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
    rule_agent: Callable[..., dict[str, Any]],
    configuration: Any | None = None,
) -> tuple[dict[str, list[Any]], ...]:
    """Build the four actions available to the neural controller."""

    player = int(observation["player"])
    hand_count = len(observation["farms"][player]["hands"])
    expert_action = normalize_action(
        call_agent(expert, observation, configuration), hand_count
    )
    rule_action = normalize_action(
        call_agent(rule_agent, observation, configuration), hand_count
    )
    return (
        copy.deepcopy(expert_action),
        copy.deepcopy(rule_action),
        {
            "farmer": copy.deepcopy(expert_action["farmer"]),
            "hands": copy.deepcopy(expert_action["hands"]),
            "market": copy.deepcopy(rule_action["market"]),
        },
        {
            "farmer": copy.deepcopy(rule_action["farmer"]),
            "hands": copy.deepcopy(rule_action["hands"]),
            "market": copy.deepcopy(expert_action["market"]),
        },
    )


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
        self.rule_agent = RuleBasedAgent()

    def __call__(
        self, observation: Any, configuration: Any | None = None
    ) -> dict[str, Any]:
        features = self.extractor.extract(observation, configuration).to_dense()
        choice = self.policy.predict(features)
        actions = candidate_actions(
            observation,
            self.expert_module.agent,
            self.rule_agent,
            configuration,
        )
        return actions[choice]
