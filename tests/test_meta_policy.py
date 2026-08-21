import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.kaggriculture.rl.meta_policy import (
    NumpyMetaPolicy,
    call_agent,
    candidate_actions,
)
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

    def test_candidates_never_change_the_expert_route(self):
        observation = {
            "player": 0,
            "farms": [{"hands": [[1, 1]]}, {"hands": []}],
            "private": {
                "shed": {"MILK": 10, "WOOL": 8},
            },
        }

        def expert(_):
            return {
                "farmer": ["PICKUP", "MILK", 3],
                "hands": [["NORTH"]],
                "market": [
                    ["BUY_PRODUCT", "WHEAT", 5],
                    ["SELL", "MILK", 2],
                ],
            }

        actions = candidate_actions(observation, expert)

        for action in actions:
            self.assertEqual(action["farmer"], ["PICKUP", "MILK", 3])
            self.assertEqual(action["hands"], [["NORTH"]])
            self.assertEqual(action["market"][0], ["BUY_PRODUCT", "WHEAT", 5])
        milk_quantities = [
            next(
                order[2]
                for order in action["market"]
                if order[:2] == ["SELL", "MILK"]
            )
            for action in actions
        ]
        self.assertEqual(milk_quantities, [2, 4, 5, 7])

    def test_rejects_old_candidate_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.npz"
            np.savez_compressed(
                path,
                w0=np.zeros((2, 2)),
                b0=np.zeros(2),
                w1=np.zeros((2, 2)),
                b1=np.zeros(2),
                w2=np.zeros((4, 2)),
                b2=np.zeros(4),
                feature_count=np.asarray(2),
                candidate_count=np.asarray(4),
            )

            with self.assertRaisesRegex(ValueError, "policy version"):
                NumpyMetaPolicy(path)


if __name__ == "__main__":
    unittest.main()
