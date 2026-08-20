"""Full ML agent: learned worker policy + learned economic policy."""

from __future__ import annotations

from pathlib import Path

from src.kaggriculture.learning.behavior_model import BehaviorCloningPolicy, commands_to_action
from src.kaggriculture.learning.economic_model import EconomicModel

from src.kaggriculture.learning.economic_guard import guard_market_orders

MODELS_DIR = Path(__file__).resolve().parent / "artifacts"

WORKER_MODEL_PATH = MODELS_DIR / "worker_bc.npz"
ECONOMIC_MODEL_PATH = MODELS_DIR / "economic_bc.npz"


WORKER_POLICY = BehaviorCloningPolicy(
    WORKER_MODEL_PATH
)

ECONOMIC_POLICY = EconomicModel(
    ECONOMIC_MODEL_PATH
)


def agent(obs):
    """
    Fully learned Kaggriculture agent.

    Workers:
        Worker Behavior Cloning policy.

    Market:
        Multi-slot Economic Behavior Cloning policy.
    """

    commands = WORKER_POLICY.predict_commands(obs)

    market = ECONOMIC_POLICY.predict_market(obs)
    market = guard_market_orders(obs, market)

    return commands_to_action(
        commands,
        market,
    )
