"""Opponent-pool acquisition and local league evaluation."""

from .evaluator import AgentSpec, evaluate_league, load_agent_file
from .notebook_source import DEFAULT_NOTEBOOKS, fetch_notebook_agents

__all__ = (
    "AgentSpec",
    "DEFAULT_NOTEBOOKS",
    "evaluate_league",
    "fetch_notebook_agents",
    "load_agent_file",
)
