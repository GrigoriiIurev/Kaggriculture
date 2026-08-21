import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.kaggriculture.rl.meta_policy import NumpyMetaPolicy, call_agent
from src.kaggriculture.rl.train_meta_controller import save_fallback_policy


class MetaPolicyTests(unittest.TestCase):
    def test_fallback_always_selects_expert(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fallback.npz"
            save_fallback_policy(path, feature_count=5, hidden_sizes=(3, 2))
            policy = NumpyMetaPolicy(path)

            self.assertEqual(policy.predict([99, -4, 2, 1, 0]), 0)

    def test_call_agent_supports_one_and_two_arguments(self):
        one = lambda observation: {"value": observation}
        two = lambda observation, configuration: {
            "value": observation,
            "configuration": configuration,
        }

        self.assertEqual(call_agent(one, 3, 4), {"value": 3})
        self.assertEqual(
            call_agent(two, 3, 4), {"value": 3, "configuration": 4}
        )


if __name__ == "__main__":
    unittest.main()
