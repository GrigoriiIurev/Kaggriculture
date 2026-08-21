import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.kaggriculture.core.action_codec import WORKER_OPERATION_TO_ID
from src.kaggriculture.core.legal_actions import legal_worker_arguments, legal_worker_operations
from src.kaggriculture.core.state_parser import parse_observation
from src.kaggriculture.data.worker_dataset import WorkerFeatureExtractor, build_worker_dataset, episode_split
from src.kaggriculture.learning.evaluate_behavior_policy import evaluate_policy


def _farm(money, farmer, hands):
    return {
        "money": money,
        "tiles": [[None, None], [None, "LOCKED"]],
        "farmer": list(farmer),
        "hands": [list(position) for position in hands],
        "unlocked_quadrants": ["NW"],
        "hires_today": len(hands),
    }


def _observation(step=10):
    return {
        "player": 0,
        "step": step,
        "day": 0,
        "hour": step,
        "farms": [
            _farm(3000, (0, 0), ((1, 0),)),
            _farm(3000, (0, 0), ()),
        ],
        "private": {
            "shed": {"WHEAT": 5},
            "seeds": {"WHEAT": 2},
            "inventories": [{}, {"WHEAT": 3}],
        },
        "market": {
            "inventory": {"WHEAT": 10000},
            "prices": {"WHEAT": 25},
        },
        "town": {"unlocked_shops": []},
    }


class WorkerDatasetTests(unittest.TestCase):
    def test_policy_evaluation_normalizes_permissive_replay_commands(self):
        episode_id = next(
            value for value in range(1, 1_000) if episode_split(value) == "holdout"
        )
        transition = {
            "episode_id": episode_id,
            "observation": _observation(),
            "action": {
                "farmer": ["FEED", "WHEAT"],
                "hands": [["PASS", "IGNORED"]],
            },
        }
        policy = Mock()
        policy.predict_commands.return_value = (["PASS"], ["PASS"])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "transitions.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as output:
                output.write(json.dumps(transition) + "\n")
            with patch(
                "src.kaggriculture.learning.evaluate_behavior_policy."
                "BehaviorCloningPolicy",
                return_value=policy,
            ):
                report = evaluate_policy(source, "unused.npz")

        self.assertEqual(report["holdout_worker_samples"], 2)

    def test_legal_mask_requires_an_item_before_pickup(self):
        raw = _observation()
        raw["private"]["shed"] = {}
        state = parse_observation(raw)

        operations = legal_worker_operations(state, 0)

        self.assertNotIn("PICKUP", operations)
        raw["private"]["shed"] = {"WHEAT": 2}
        state = parse_observation(raw)
        self.assertIn("PICKUP", legal_worker_operations(state, 0))
        self.assertEqual(legal_worker_arguments(state, 0, "PICKUP"), {"WHEAT"})

    def test_legal_mask_allows_harvest_from_animal(self):
        raw = _observation()
        raw["farms"][0]["tiles"][0][0] = {
            "kind": "PASTURE",
            "animal": "COW",
            "placed_day": 0,
            "yield_units": 2,
            "fed_today": False,
            "consecutive_unfed": 0,
            "cared_today": False,
            "fertilizer_available": False,
            "pending_care_bonus": 0,
        }
        state = parse_observation(raw)

        self.assertIn("HARVEST", legal_worker_operations(state, 0))

    def test_legal_mask_rejects_place_on_occupied_structure(self):
        raw = _observation()
        for farm in raw["farms"]:
            farm["tiles"] = [[None for _ in range(10)] for _ in range(10)]
        raw["private"]["inventories"][0] = {"COW": 1}
        raw["farms"][0]["tiles"][0][0] = {
            "kind": "PASTURE",
            "animal": "COW",
            "placed_day": 0,
            "yield_units": 0,
            "fed_today": True,
            "consecutive_unfed": 0,
            "cared_today": True,
            "fertilizer_available": False,
            "pending_care_bonus": 0,
        }
        state = parse_observation(raw)

        self.assertNotIn("PLACE", legal_worker_operations(state, 0))

    def test_worker_context_focuses_position_and_inventory(self):
        extractor = WorkerFeatureExtractor(board_size=2)
        farmer = extractor.extract(_observation(), 0)
        hand = extractor.extract(_observation(), 1)
        farmer_named = {
            extractor.feature_name(index): value
            for index, value in zip(farmer.indices, farmer.values)
        }
        hand_named = {
            extractor.feature_name(index): value
            for index, value in zip(hand.indices, hand.values)
        }

        self.assertEqual(farmer_named["focused_worker[0,0]"], 1)
        self.assertEqual(hand_named["focused_worker[0,1]"], 1)
        self.assertNotIn("worker_inventory_wheat_log", farmer_named)
        self.assertGreater(hand_named["worker_inventory_wheat_log"], 0)
        self.assertEqual(farmer.size, hand.size)

    def test_episode_split_is_stable(self):
        first = episode_split(123456)

        self.assertEqual(first, episode_split(123456))
        self.assertIn(first, {"train", "holdout"})

    def test_builds_group_with_one_target_per_worker(self):
        transition = {
            "episode_id": 123456,
            "episode_type": "public",
            "seat": 0,
            "step": 10,
            "observation": _observation(),
            "next_observation": _observation(step=11),
            "action": {
                "farmer": ["EAST"],
                "hands": [["PICKUP", "WHEAT", 1]],
                "market": [],
            },
            "final_reward": 100,
            "margin": 20,
            "outcome": "win",
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "transitions.jsonl.gz"
            output = Path(directory) / "worker_dataset.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(transition) + "\n")

            manifest = build_worker_dataset(source, output, board_size=2)
            with gzip.open(output, "rt", encoding="utf-8") as stream:
                record = json.loads(stream.readline())
            action_schema = json.loads(
                (Path(directory) / "action_schema.json").read_text()
            )

        self.assertEqual(manifest["transition_groups"], 1)
        self.assertEqual(manifest["worker_samples"], 2)
        self.assertEqual(len(record["workers"]), 2)
        self.assertEqual(
            record["workers"][0]["target"]["operation_id"],
            WORKER_OPERATION_TO_ID["EAST"],
        )
        self.assertEqual(action_schema["worker_operations"][0], "PASS")

    def test_normalizes_extra_arguments_accepted_by_game_engine(self):
        transition = {
            "episode_id": 123457,
            "episode_type": "public",
            "seat": 0,
            "step": 10,
            "observation": _observation(),
            "next_observation": _observation(step=11),
            "action": {
                "farmer": ["PASS", "IGNORED"],
                "hands": [["FEED", "WHEAT"]],
                "market": [],
            },
            "final_reward": 100,
            "margin": 20,
            "outcome": "win",
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "transitions.jsonl.gz"
            output = Path(directory) / "worker_dataset.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(transition) + "\n")

            manifest = build_worker_dataset(source, output, board_size=2)
            with gzip.open(output, "rt", encoding="utf-8") as stream:
                record = json.loads(stream.readline())

        self.assertEqual(manifest["normalized_recorded_commands"], 2)
        self.assertEqual(
            record["workers"][1]["target"]["operation_id"],
            WORKER_OPERATION_TO_ID["FEED"],
        )


if __name__ == "__main__":
    unittest.main()
