"""Event-driven Gymnasium environment for league market training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core.game_data import BASE_PRICES
from ..league.evaluator import load_agent_file
from .league_policy import (
    ACTION_DIMS,
    RESIDUAL_PRODUCTS,
    MarketHistoryFeatures,
    apply_market_residual,
    enforce_endgame_liquidation,
    market_decision_available,
    premium_sell_quantities,
)
from .meta_policy import call_agent

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised in Colab
    raise ImportError("Training requires gymnasium; install requirements-rl.txt") from exc


class KaggricultureLeagueEnv(gym.Env):
    """Train only when the policy can actually change a premium sale."""

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
        win_bonus: float = 10.0,
        market_reward_scale: float = 1_000.0,
    ) -> None:
        super().__init__()
        if not opponent_paths:
            raise ValueError("At least one league opponent is required")
        self.incumbent_path = Path(incumbent_path)
        self.opponent_paths = tuple(Path(path) for path in opponent_paths)
        weights = np.asarray(
            opponent_weights
            if opponent_weights is not None
            else np.ones(len(opponent_paths)),
            dtype=np.float64,
        )
        if weights.shape != (len(self.opponent_paths),) or np.any(weights < 0):
            raise ValueError("Opponent weights must be non-negative and match the pool")
        if float(weights.sum()) <= 0:
            raise ValueError("At least one opponent weight must be positive")
        if market_reward_scale <= 0:
            raise ValueError("market_reward_scale must be positive")
        self.opponent_weights = weights / weights.sum()
        self.episode_steps = episode_steps
        self.seed_offset = seed_offset
        self.fixed_opponent = fixed_opponent
        self.fixed_seat = fixed_seat
        self.win_bonus = win_bonus
        self.market_reward_scale = market_reward_scale
        self.features = MarketHistoryFeatures()
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.features.feature_count,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(
            np.asarray(ACTION_DIMS, dtype=np.int64)
        )
        self._episode_index = 0
        self._states: list[Any] | None = None
        self._environment: Any = None
        self._learner_seat = 0
        self._opponent_index = 0
        self._previous_margin = 0.0
        self._pending_terminal = False
        self._cached_observation: dict[str, Any] | None = None
        self._cached_base_action: dict[str, Any] | None = None
        self._cached_opponent_action: dict[str, Any] | None = None

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
        self._pending_terminal = False
        observation, terminated, skipped = self._advance_to_decision()
        self._pending_terminal = terminated
        self._previous_margin = self._money_margin()
        info = self._info(game_seed)
        info.update({"auto_turns": skipped, "decision_available": not terminated})
        return observation, info

    def step(
        self, action: Sequence[int] | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._states is None:
            raise RuntimeError("reset() must be called before step()")
        if self._pending_terminal:
            return self._zero_observation(), 0.0, True, False, self._terminal_info()
        raw_choices = np.asarray(action, dtype=np.int64)
        if not self.action_space.contains(raw_choices):
            raise ValueError(f"Invalid residual action {raw_choices.tolist()}")
        assert self._cached_observation is not None
        assert self._cached_base_action is not None
        assert self._cached_opponent_action is not None

        choices = np.asarray(
            enforce_endgame_liquidation(
                self._cached_observation, raw_choices
            ),
            dtype=np.int64,
        )
        base_sales = premium_sell_quantities(self._cached_base_action)
        learner_action = apply_market_residual(
            self._cached_base_action, self._cached_observation, choices
        )
        selected_sales = premium_sell_quantities(learner_action)
        changes = {
            product: int(selected_sales[product] - base_sales[product])
            for product in RESIDUAL_PRODUCTS
        }
        effective = any(changes.values())
        quality_reward = self._market_quality_reward(selected_sales)
        opponent_seat = 1 - self._learner_seat
        actions = [None, None]
        actions[self._learner_seat] = learner_action
        actions[opponent_seat] = self._cached_opponent_action
        self._states = self._environment.step(actions)
        self.features.record_action(choices)

        observation, terminated, skipped = self._advance_to_decision()
        self._pending_terminal = terminated
        margin = self._money_margin()
        money_reward = (margin - self._previous_margin) / 5000.0
        self._previous_margin = margin
        reward = money_reward + quality_reward
        info = self._info(None)
        info.update(
            {
                "choices": choices.tolist(),
                "raw_choices": raw_choices.tolist(),
                "safety_forced": bool(np.any(choices != raw_choices)),
                "residual_effective": effective,
                "sale_quantity_changes": changes,
                "market_quality_reward": quality_reward,
                "money_reward": money_reward,
                "auto_turns": skipped,
            }
        )
        if terminated:
            terminal = self._terminal_info()
            reward += self.win_bonus * int(terminal["outcome"])
            reward -= min(5.0, float(terminal["terminal_premium_stock"]) / 20.0)
            info.update(terminal)
        return observation, float(reward), terminated, False, info

    def _advance_to_decision(self) -> tuple[np.ndarray, bool, int]:
        """Play incumbent turns until a premium sale can really be changed."""

        skipped = 0
        while not self._terminated():
            learner_obs = self._observation_for_seat(self._learner_seat)
            opponent_seat = 1 - self._learner_seat
            opponent_obs = self._observation_for_seat(opponent_seat)
            base_action = call_agent(
                self._incumbent, learner_obs, self._environment.configuration
            )
            opponent_action = call_agent(
                self._opponent, opponent_obs, self._environment.configuration
            )
            features = self.features.extract(learner_obs, base_action)
            if market_decision_available(base_action, learner_obs):
                self._cached_observation = learner_obs
                self._cached_base_action = base_action
                self._cached_opponent_action = opponent_action
                return features, False, skipped
            actions = [None, None]
            actions[self._learner_seat] = base_action
            actions[opponent_seat] = opponent_action
            self._states = self._environment.step(actions)
            skipped += 1
        self._cached_observation = None
        self._cached_base_action = None
        self._cached_opponent_action = None
        return self._zero_observation(), True, skipped

    def _market_quality_reward(self, sales: dict[str, int]) -> float:
        assert self._cached_observation is not None
        prices = self._cached_observation.get("market", {}).get("prices", {}) or {}
        raw = 0.0
        for product in RESIDUAL_PRODUCTS:
            relative_price = float(prices.get(product, 0) or 0) / max(
                1.0, float(BASE_PRICES[product])
            )
            raw += int(sales.get(product, 0)) * max(0.0, relative_price - 1.0)
        return float(np.clip(raw / self.market_reward_scale, 0.0, 0.1))

    def _terminated(self) -> bool:
        assert self._states is not None
        return str(self._states[self._learner_seat].status) != "ACTIVE"

    def _zero_observation(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

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

    def _current_turn(self) -> int:
        assert self._states is not None
        return int(self._states[0].observation.get("step", 0) or 0)

    def _terminal_info(self) -> dict[str, Any]:
        margin = self._money_margin()
        outcome = 1 if margin > 0 else -1 if margin < 0 else 0
        return {
            "outcome": outcome,
            "win": outcome > 0,
            "tie": outcome == 0,
            "learner_money": self._money(self._learner_seat),
            "opponent_money": self._money(1 - self._learner_seat),
            "money_margin": margin,
            "terminal_premium_stock": self._premium_stock(),
        }

    def _premium_stock(self) -> int:
        observation = self._observation_for_seat(self._learner_seat)
        shed = observation.get("private", {}).get("shed", {}) or {}
        return sum(max(0, int(shed.get(product, 0) or 0)) for product in RESIDUAL_PRODUCTS)

    def _info(self, game_seed: int | None) -> dict[str, Any]:
        info = {
            "money_margin": self._money_margin(),
            "game_turn": self._current_turn(),
            "learner_seat": self._learner_seat,
            "opponent_index": self._opponent_index,
            "opponent_path": str(self.opponent_paths[self._opponent_index]),
        }
        if game_seed is not None:
            info["game_seed"] = game_seed
        return info
