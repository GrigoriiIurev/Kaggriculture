"""Stable production agent assembled from the rule-based planners."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from .core.state_parser import parse_observation
from .planning.economic_planner import EconomicConfig, EconomicPlanner
from .planning.task_generator import TaskGenerator
from .planning.worker_planner import WorkerPlanner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMOTED_CONFIG_PATH = PROJECT_ROOT / "artifacts/models/promoted_economic_config.json"
PROMOTED_WORKER_MODEL_PATH = PROJECT_ROOT / "artifacts/models/promoted_worker_bc.npz"


def load_economic_config(path: str | Path | None = None) -> EconomicConfig:
    """Load a promoted configuration, falling back to the proven baseline."""

    config_path = Path(path) if path is not None else PROMOTED_CONFIG_PATH
    if not config_path.exists():
        return EconomicConfig()

    with config_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    values = payload.get("config", payload)
    allowed = {item.name for item in fields(EconomicConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown economic config fields: {sorted(unknown)}")
    return EconomicConfig(**values)


class RuleBasedAgent:
    """One isolated agent instance, suitable for games and local tournaments."""

    def __init__(self, config: EconomicConfig | None = None) -> None:
        self.task_generator = TaskGenerator()
        self.economic_planner = EconomicPlanner(config or EconomicConfig())
        self.worker_planner = WorkerPlanner()

    def __call__(
        self,
        observation: Any,
        configuration: Any | None = None,
    ) -> dict[str, Any]:
        state = parse_observation(observation)
        if state.step == 0:
            self.worker_planner.reset()

        farm_tasks = self.task_generator.generate(state)
        economy = self.economic_planner.plan(state, farm_tasks)
        workers = self.worker_planner.plan(state, (*farm_tasks, *economy.tasks))
        action = workers.action
        action["market"] = [list(order) for order in economy.market_orders]
        return action


class BehaviorCloningAgent:
    """Learned worker policy combined with the proven economic planner."""

    def __init__(
        self,
        model_path: str | Path,
        config: EconomicConfig | None = None,
    ) -> None:
        from .learning.behavior_model import (
            BehaviorCloningPolicy,
            commands_to_action,
        )

        self.policy = BehaviorCloningPolicy(model_path)
        self.commands_to_action = commands_to_action
        self.task_generator = TaskGenerator()
        self.economic_planner = EconomicPlanner(config or EconomicConfig())

    def __call__(
        self,
        observation: Any,
        configuration: Any | None = None,
    ) -> dict[str, Any]:
        state = parse_observation(observation)
        farm_tasks = self.task_generator.generate(state)
        economy = self.economic_planner.plan(state, farm_tasks)
        commands = self.policy.predict_commands(observation)
        market = [list(order) for order in economy.market_orders]
        return self.commands_to_action(commands, market)


def build_production_agent(
    worker_model_path: str | Path | None = None,
) -> RuleBasedAgent | BehaviorCloningAgent:
    if worker_model_path is not None:
        model_path = Path(worker_model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Worker model not found: {model_path}")
        return BehaviorCloningAgent(model_path, load_economic_config())
    if PROMOTED_WORKER_MODEL_PATH.is_file():
        return BehaviorCloningAgent(
            PROMOTED_WORKER_MODEL_PATH, load_economic_config()
        )
    return RuleBasedAgent(load_economic_config())
