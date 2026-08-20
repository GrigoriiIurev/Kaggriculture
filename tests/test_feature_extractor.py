import gzip
import json
import tempfile
import unittest
from pathlib import Path

from src.kaggriculture.data.feature_extractor import FeatureExtractor, build_feature_dataset


def _farm(money, tiles, farmer=(0, 0), hands=()):
    return {
        "money": money,
        "tiles": tiles,
        "farmer": list(farmer),
        "hands": [list(position) for position in hands],
        "unlocked_quadrants": ["NW"],
        "hires_today": len(hands),
    }


def _observation(step=50):
    plant = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 1,
        "watered_today": False,
        "consecutive_unwatered": 1,
        "yield_units": 2,
        "max_lifespan_step": 120,
        "fertilized_until_day": -1,
    }
    cow = {
        "kind": "PASTURE",
        "animal": "COW",
        "placed_day": 1,
        "yield_units": 1,
        "fed_today": True,
        "consecutive_unfed": 0,
        "cared_today": False,
        "fertilizer_available": True,
        "pending_care_bonus": 2,
    }
    return {
        "player": 0,
        "step": step,
        "day": 2,
        "hour": 2,
        "farms": [
            _farm(1200, [[plant, cow], [None, "LOCKED"]], hands=((1, 0),)),
            _farm(900, [[None, None], [None, "LOCKED"]]),
        ],
        "private": {
            "shed": {"WHEAT": 20, "MILK": 3},
            "seeds": {"WHEAT": 4},
            "inventories": [{"WHEAT": 1}, {}],
        },
        "market": {
            "inventory": {"WHEAT": 9980, "MILK": 10010},
            "prices": {"WHEAT": 50, "MILK": 160},
        },
        "town": {"unlocked_shops": ["BAKERY", "YARN_STORE"]},
    }


class FeatureExtractorTests(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor(board_size=2)

    def _named_values(self, features):
        return {
            self.extractor.feature_name(index): value
            for index, value in zip(features.indices, features.values)
        }

    def test_extracts_scalar_and_spatial_features(self):
        features = self.extractor.extract(_observation())
        named = self._named_values(features)

        self.assertEqual(features.size, self.extractor.feature_count)
        self.assertEqual(named["me.crop_wheat[0,0]"], 1)
        self.assertEqual(named["me.animal_cow[0,1]"], 1)
        self.assertEqual(named["me.hand_count[0,1]"], 1)
        self.assertEqual(named["opponent.locked[1,1]"], 1)
        self.assertEqual(named["market_price_ratio_wheat"], 2)
        self.assertGreater(named["shed_wheat_log"], 0)

    def test_vector_layout_is_stable_when_values_change(self):
        first = self.extractor.extract(_observation())
        changed = _observation(step=51)
        changed["farms"][0]["money"] = 5000
        second = self.extractor.extract(changed)

        self.assertEqual(first.size, second.size)
        self.assertEqual(self.extractor.schema()["feature_count"], first.size)
        self.assertEqual(first.indices, tuple(sorted(first.indices)))
        self.assertEqual(second.indices, tuple(sorted(second.indices)))

    def test_terminal_observation_has_no_remaining_season(self):
        features = self.extractor.extract(_observation(step=719))
        named = self._named_values(features)

        self.assertNotIn("remaining_fraction", named)

    def test_dense_conversion_preserves_sparse_values(self):
        features = self.extractor.extract(_observation())
        dense = features.to_dense()

        self.assertEqual(len(dense), features.size)
        for index, value in zip(features.indices, features.values):
            self.assertEqual(dense[index], value)

    def test_rejects_wrong_board_size(self):
        with self.assertRaisesRegex(ValueError, "expects 10x10"):
            FeatureExtractor(board_size=10).extract(_observation())

    def test_builds_feature_dataset(self):
        observation = _observation()
        transition = {
            "episode_id": 123,
            "episode_type": "public",
            "seat": 0,
            "step": 50,
            "observation": observation,
            "action": {"farmer": ["WATER"], "hands": [["PASS"]], "market": []},
            "next_observation": _observation(step=51),
            "final_reward": 100,
            "margin": 20,
            "outcome": "win",
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "transitions.jsonl.gz"
            output = Path(directory) / "features.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(transition) + "\n")

            manifest = build_feature_dataset(source, output, board_size=2)
            with gzip.open(output, "rt", encoding="utf-8") as stream:
                record = json.loads(stream.readline())
            schema = json.loads((Path(directory) / "feature_schema.json").read_text())

        self.assertEqual(manifest["records"], 1)
        self.assertEqual(record["action"]["farmer"], ["WATER"])
        self.assertFalse(record["terminal"])
        self.assertEqual(len(record["feature_indices"]), len(record["feature_values"]))
        self.assertEqual(schema["feature_count"], manifest["feature_count"])


if __name__ == "__main__":
    unittest.main()
