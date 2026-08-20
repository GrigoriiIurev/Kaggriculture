"""Hybrid agent: learned worker policy with the existing rule-based economy."""

from __future__ import annotations

from pathlib import Path

from src.kaggriculture.learning.behavior_model import BehaviorCloningPolicy, commands_to_action
from src.kaggriculture.planning.economic_planner import EconomicPlanner
from src.kaggriculture.core.state_parser import parse_observation
from src.kaggriculture.planning.task_generator import TaskGenerator


MODEL_PATH = Path(__file__).resolve().parent / "artifacts/worker_bc.npz"
POLICY = BehaviorCloningPolicy(MODEL_PATH)
TASK_GENERATOR = TaskGenerator()
ECONOMIC_PLANNER = EconomicPlanner()


def agent(obs):
    state = parse_observation(obs)
    farm_tasks = TASK_GENERATOR.generate(state)
    economy = ECONOMIC_PLANNER.plan(state, farm_tasks)
    commands = POLICY.predict_commands(obs)
    market = [list(order) for order in economy.market_orders]
    return commands_to_action(commands, market)
