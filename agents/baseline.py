"""Unmodified baseline used as the control group in tournaments."""

from src.kaggriculture.agent import RuleBasedAgent
from src.kaggriculture.planning.economic_planner import EconomicConfig


agent = RuleBasedAgent(EconomicConfig())
