"""Gymnasium environment for training an expert-gated Kaggriculture policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..agent import RuleBasedAgent
from ..data.feature_extractor import FeatureExtractor
from .meta_policy import call_agent, candidate_actions, load_agent_module

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised in Colab
    raise ImportError(
        "Training requires gymnasium; install requirements-rl.txt"
    ) from exc


class KaggricultureMetaEnv(gym.Env):
    """Discrete controller that plays against an independent expert copy."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        expert_path: str | Path,
        episode_steps: int = 720,
        seed_offset: int = 0,
        fixed_seats: bool = False,
        win_bonus: float = 5.0,
    ) -> None:
        super().__init__()
        self.expert_path = Path(expert_path)
        self.episode_steps = episode_steps
        self.seed_offset = seed_offset
        self.fixed_seats = fixed_seats
        self.win_bonus = win_bonus
        self.extractor = FeatureExtractor()
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.extractor.feature_count,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)
        self._episode_index = 0
        self._states: list[Any] | None = None
        self._environment: Any = None
        self._learner_seat = 0
        self._previous_margin = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        from kaggle_environments import make

        if seed is None:
            game_seed = self.seed_offset + self._episode_index
        else:
            game_seed = self.seed_offset + int(seed)
        if self.fixed_seats:
            self._learner_seat = self._episode_index % 2
        else:
            self._learner_seat = int(self.np_random.integers(0, 2))
        self._episode_index += 1
        self._environment = make(
            "kaggriculture",
            configuration={
                "episodeSteps": self.episode_steps,
                "seed": game_seed,
            },
            debug=False,
        )
        self._states = self._environment.reset(2)
        self._learner_expert = load_agent_module(
            self.expert_path, "kaggriculture_training_learner_expert"
        ).agent
        self._opponent = load_agent_module(
            self.expert_path, "kaggriculture_training_opponent_expert"
        ).agent
        self._rule_agent = RuleBasedAgent()
        self._previous_margin = self._money_margin()
        return self._observation(), {
            "game_seed": game_seed,
            "learner_seat": self._learner_seat,
        }

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._states is None:
            raise RuntimeError("reset() must be called before step()")
        choice = int(action)
        if not self.action_space.contains(choice):
            raise ValueError(f"Invalid candidate {choice}")

        learner_obs = self._observation_for_seat(self._learner_seat)
        opponent_seat = 1 - self._learner_seat
        opponent_obs = self._observation_for_seat(opponent_seat)
        candidates = candidate_actions(
            learner_obs,
            self._learner_expert,
            self._rule_agent,
            self._environment.configuration,
        )
        opponent_action = call_agent(
            self._opponent,
            opponent_obs, self._environment.configuration
        )
        actions = [None, None]
        actions[self._learner_seat] = candidates[choice]
        actions[opponent_seat] = opponent_action
        self._states = self._environment.step(actions)

        terminated = str(self._states[self._learner_seat].status) != "ACTIVE"
        margin = self._money_margin()
        reward = (margin - self._previous_margin) / 1000.0
        self._previous_margin = margin
        info: dict[str, Any] = {
            "money_margin": margin,
            "learner_seat": self._learner_seat,
        }
        if terminated:
            outcome = 1 if margin > 0 else -1 if margin < 0 else 0
            reward += self.win_bonus * outcome
            info.update(
                {
                    "win": outcome > 0,
                    "tie": outcome == 0,
                    "outcome": outcome,
                    "learner_money": self._money(self._learner_seat),
                    "opponent_money": self._money(1 - self._learner_seat),
                }
            )
            observation = np.zeros(
                self.observation_space.shape, dtype=np.float32
            )
        else:
            observation = self._observation()
        return observation, float(reward), terminated, False, info

    def _observation(self) -> np.ndarray:
        assert self._states is not None
        raw = self._observation_for_seat(self._learner_seat)
        return np.asarray(
            self.extractor.extract(raw, self._environment.configuration).to_dense(),
            dtype=np.float32,
        )

    def _observation_for_seat(self, seat: int) -> dict[str, Any]:
        """Materialize shared fields omitted from non-primary state entries."""

        assert self._states is not None
        observation = dict(self._states[seat].observation)
        primary = self._states[0].observation
        if "step" not in observation:
            observation["step"] = int(primary["step"])
        observation["player"] = seat
        return observation

    def _money(self, seat: int) -> float:
        assert self._states is not None
        farms = self._states[0].observation["farms"]
        return float(farms[seat]["money"])

    def _money_margin(self) -> float:
        return self._money(self._learner_seat) - self._money(1 - self._learner_seat)
