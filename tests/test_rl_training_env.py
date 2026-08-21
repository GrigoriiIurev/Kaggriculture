import tempfile
import unittest
from pathlib import Path

from src.kaggriculture.rl.training_env import KaggricultureMetaEnv


class RlTrainingEnvTests(unittest.TestCase):
    def test_shared_step_is_materialized_for_second_seat(self):
        with tempfile.TemporaryDirectory() as directory:
            expert = Path(directory) / "expert.py"
            expert.write_text(
                "def agent(obs):\n"
                "    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n",
                encoding="utf-8",
            )
            environment = KaggricultureMetaEnv(
                expert, episode_steps=4, fixed_seats=True
            )
            environment.reset(seed=1)
            environment._learner_seat = 1

            observation = environment._observation_for_seat(1)

            self.assertEqual(observation["step"], 0)
            self.assertEqual(observation["player"], 1)


if __name__ == "__main__":
    unittest.main()
