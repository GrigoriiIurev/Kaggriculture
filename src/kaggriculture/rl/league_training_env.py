"""Gymnasium environment for a market-only policy trained against a league."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..league.evaluator import load_agent_file
from .league_policy import ACTION_DIMS, MarketHistoryFeatures, apply_market_residual
from .meta_policy import call_agent

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised in Colab
    raise ImportError("Training requires gymnasium; install requirements-rl.txt") from exc


class KaggricultureLeagueEnv(gym.Env):
    """Keep the incumbent route fixed and learn only additional premium sales."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        incumbent_path: str | Path,
        opponent_paths: Sequence[str | Path],
        opponent_weights: Sequence[float] | None = None,
        *,
        episode_steps: int = 720,
        seed_offset: int = 0,
        fixed_opponent: int | None = None,
        fixed_seat: int | None = None,
        win_bonus: float = 20.0,
    ) -> None:
        super().__init__()
        if not opponent_paths:
            raise ValueError("At least one league opponent is required")
        self.incumbent_path = Path(incumbent_path)
        self.opponent_paths = tuple(Path(path) for path in opponent_paths)
        weights = np.asarray(
            opponent_weights if opponent_weights is not None else np.ones(len(opponent_paths)),
            dtype=np.float64,
        )
        if weights.shape != (len(self.opponent_paths),) or np.any(weights < 0):
            raise ValueError("Opponent weights must be non-negative and match the pool")
        if float(weights.sum()) <= 0:
            raise ValueError("At least one opponent weight must be positive")
        self.opponent_weights = weights / weights.sum()
        self.episode_steps = episode_steps
        self.seed_offset = seed_offset
        self.fixed_opponent = fixed_opponent
        self.fixed_seat = fixed_seat
        self.win_bonus = win_bonus
        self.features = MarketHistoryFeatures()
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.features.feature_count,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(np.asarray(ACTION_DIMS, dtype=np.int64))
        self._episode_index = 0
        self._states: list[Any] | None = None
        self._environment: Any = None
        self._learner_seat = 0
        self._opponent_index = 0
        self._previous_margin = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        from kaggle_environments import make

        options = options or {}
        game_number = self._episode_index if seed is None else int(seed)
        game_seed = self.seed_offset + game_number
        opponent_index = options.get("opponent_index", self.fixed_opponent)
        if opponent_index is None:
            opponent_index = int(
                self.np_random.choice(len(self.opponent_paths), p=self.opponent_weights)
            )
        learner_seat = options.get("learner_seat", self.fixed_seat)
        if learner_seat is None:
            learner_seat = self._episode_index % 2
        self._opponent_index = int(opponent_index)
        self._learner_seat = int(learner_seat)
        if not 0 <= self._opponent_index < len(self.opponent_paths):
            raise ValueError("Invalid opponent index")
        if self._learner_seat not in (0, 1):
            raise ValueError("learner_seat must be 0 or 1")
        self._episode_index += 1

        self._environment = make(
            "kaggriculture",
            configuration={"episodeSteps": self.episode_steps, "seed": game_seed},
            debug=False,
        )
        self._states = self._environment.reset(2)
        self._incumbent = load_agent_file(self.incumbent_path)
        self._opponent = load_agent_file(self.opponent_paths[self._opponent_index])
        self.features.reset()
        self._previous_margin = self._money_margin()
        return self._observation(), self._info(game_seed)

    def step(
        self, action: Sequence[int] | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._states is None:
            raise RuntimeError("reset() must be called before step()")
        choices = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(choices):
            raise ValueError(f"Invalid residual action {choices.tolist()}")

        learner_obs = self._observation_for_seat(self._learner_seat)
        opponent_seat = 1 - self._learner_seat
        opponent_obs = self._observation_for_seat(opponent_seat)
        base_action = call_agent(
            self._incumbent, learner_obs, self._environment.configuration
        )
        learner_action = apply_market_residual(base_action, learner_obs, choices)
        opponent_action = call_agent(
            self._opponent, opponent_obs, self._environment.configuration
        )
        actions = [None, None]
        actions[self._learner_seat] = learner_action
        actions[opponent_seat] = opponent_action
        self._states = self._environment.step(actions)
        self.features.record_action(choices)

        terminated = str(self._states[self._learner_seat].status) != "ACTIVE"
        margin = self._money_margin()
        reward = (margin - self._previous_margin) / 1000.0
        self._previous_margin = margin
        info = self._info(None)
        if terminated:
            outcome = 1 if margin > 0 else -1 if margin < 0 else 0
            reward += self.win_bonus * outcome
            info.update(
                {
                    "outcome": outcome,
                    "win": outcome > 0,
                    "tie": outcome == 0,
                    "learner_money": self._money(self._learner_seat),
                    "opponent_money": self._money(opponent_seat),
                }
            )
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = self._observation()
        return observation, float(reward), terminated, False, info

    def _observation(self) -> np.ndarray:
        return self.features.extract(self._observation_for_seat(self._learner_seat))

    def _observation_for_seat(self, seat: int) -> dict[str, Any]:
        assert self._states is not None
        observation = dict(self._states[seat].observation)
        primary = self._states[0].observation
        observation.setdefault("step", int(primary["step"]))
        observation["player"] = seat
        return observation

    def _money(self, seat: int) -> float:
        assert self._states is not None
        return float(self._states[0].observation["farms"][seat]["money"])

    def _money_margin(self) -> float:
        return self._money(self._learner_seat) - self._money(1 - self._learner_seat)

    def _info(self, game_seed: int | None) -> dict[str, Any]:
        info = {
            "money_margin": self._money_margin(),
            "learner_seat": self._learner_seat,
            "opponent_index": self._opponent_index,
            "opponent_path": str(self.opponent_paths[self._opponent_index]),
        }
        if game_seed is not None:
            info["game_seed"] = game_seed
        return info
