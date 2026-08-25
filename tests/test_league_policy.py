import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.kaggriculture.rl.league_policy import (
    ACTION_DIMS,
    FEATURE_NAMES,
    MarketHistoryFeatures,
    NumpyLeaguePolicy,
    apply_market_residual,
)
from src.kaggriculture.rl.train_league_controller import (
    promotion_gate,
    save_fallback_policy,
)


def observation(step=0):
    empty = [[None for _ in range(10)] for _ in range(10)]
    farm = {
        "money": 3000,
        "tiles": empty,
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW"],
    }
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [farm, farm],
        "private": {
            "shed": {"MILK": 10, "WOOL": 8},
            "inventories": [{}],
        },
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }


class LeaguePolicyTests(unittest.TestCase):
    def test_fallback_selects_zero_for_every_product(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fallback.npz"
            save_fallback_policy(path, hidden_sizes=(3, 2))
            policy = NumpyLeaguePolicy(path)

            np.testing.assert_array_equal(
                policy.predict(np.zeros(len(FEATURE_NAMES))),
                np.zeros(len(ACTION_DIMS), dtype=np.int64),
            )

    def test_features_include_history_without_changing_shape(self):
        extractor = MarketHistoryFeatures()
        first = extractor.extract(observation(0))
        second_obs = observation(4)
        second_obs["market"]["prices"]["MILK"] = 200
        second = extractor.extract(second_obs)

        self.assertEqual(first.shape, (len(FEATURE_NAMES),))
        self.assertEqual(second.shape, first.shape)
        self.assertFalse(np.array_equal(first, second))

    def test_residual_preserves_route_and_existing_market_order(self):
        obs = observation()
        base = {
            "farmer": ["PICKUP", "MILK", 3],
            "hands": [],
            "market": [["BUY_PRODUCT", "WHEAT", 2], ["SELL", "MILK", 2]],
        }

        result = apply_market_residual(base, obs, [4, 3, 0, 0])

        self.assertEqual(result["farmer"], base["farmer"])
        self.assertEqual(result["market"][0], base["market"][0])
        self.assertEqual(result["market"][1], ["SELL", "MILK", 7])
        self.assertEqual(base["market"][1], ["SELL", "MILK", 2])

        held = apply_market_residual(base, obs, [1, 0, 0, 0])
        self.assertNotIn(["SELL", "MILK", 2], held["market"])

    def test_gate_rejects_veto_collapse(self):
        baseline = {
            "score_rate": 0.5,
            "mean_money_margin": 0.0,
            "opponents": [
                {"slug": "hard", "veto": True, "score_rate": 0.75, "mean_money_margin": 100.0}
            ],
        }
        candidate = {
            "score_rate": 0.6,
            "mean_money_margin": 500.0,
            "errors": 0,
            "opponents": [
                {"slug": "hard", "veto": True, "score_rate": 0.25, "mean_money_margin": -6000.0}
            ],
        }

        self.assertFalse(promotion_gate(baseline, candidate)["passed"])


if __name__ == "__main__":
    unittest.main()
