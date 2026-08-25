import unittest
from pathlib import Path

import numpy as np

from src.kaggriculture.core.game_data import BASE_PRICES
from src.kaggriculture.rl.league_training_env import KaggricultureLeagueEnv


class LeagueTrainingEnvTests(unittest.TestCase):
    def test_skips_empty_turns_and_reports_a_real_market_change(self):
        agent = Path("agents/baseline.py").resolve()
        environment = KaggricultureLeagueEnv(
            agent,
            [agent],
            episode_steps=200,
            fixed_seat=0,
        )

        observation, reset_info = environment.reset(seed=123)
        _, _, done, _, info = environment.step(
            np.ones(4, dtype=np.int64)
        )

        self.assertEqual(observation.shape, environment.observation_space.shape)
        self.assertGreater(reset_info["auto_turns"], 0)
        self.assertFalse(done)
        self.assertTrue(info["residual_effective"])
        self.assertTrue(any(info["sale_quantity_changes"].values()))

    def test_cheap_sale_is_not_punished_and_premium_sale_gets_small_bonus(self):
        agent = Path("agents/baseline.py").resolve()
        environment = KaggricultureLeagueEnv(
            agent, [agent], episode_steps=200, fixed_seat=0
        )
        environment.reset(seed=123)
        environment._cached_observation["market"]["prices"]["WOOL"] = 1
        cheap = environment._market_quality_reward({"WOOL": 4})
        environment._cached_observation["market"]["prices"]["WOOL"] = (
            2 * BASE_PRICES["WOOL"]
        )
        premium = environment._market_quality_reward({"WOOL": 4})

        self.assertEqual(cheap, 0.0)
        self.assertGreater(premium, 0.0)
        self.assertLessEqual(premium, 0.5)


if __name__ == "__main__":
    unittest.main()
