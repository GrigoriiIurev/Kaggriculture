import unittest
from pathlib import Path

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
