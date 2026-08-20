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


def build_production_agent() -> RuleBasedAgent:
    return RuleBasedAgent(load_economic_config())
