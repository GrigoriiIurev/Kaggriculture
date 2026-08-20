import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from src.kaggriculture.data.outcome_logger import build_dataset, summarize_trajectory
from src.kaggriculture.data.replay_parser import infer_team_name, iter_transitions, load_replay
from src.kaggriculture.data.teacher_dataset import build_teacher_transitions


def _observation(player, step, money, *, include_step=True):
    empty_tiles = [[None, None], [None, None]]
    observation = {
        "player": player,
        "day": 0,
        "hour": step,
        "farms": [
            {
                "money": money if player == 0 else 80,
                "tiles": empty_tiles,
                "farmer": [0, 0],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
            {
                "money": 80 if player == 0 else money,
                "tiles": empty_tiles,
                "farmer": [0, 0],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
        ],
        "private": {
            "shed": {"COW": 0},
            "seeds": {"WHEAT": 2},
            "inventories": [{}],
        },
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
        "remainingOverageTime": 59.9,
    }
    if include_step:
        observation["step"] = step
    return observation


def _replay(episode_id=123, teams=("Us", "Them")):
    actions = [
        ({"farmer": ["PASS"], "hands": [], "market": []},) * 2,
        (
            {"farmer": ["EAST"], "hands": [], "market": [["BUY_SEED", "WHEAT", 2]]},
            {"farmer": ["WEST"], "hands": [], "market": []},
        ),
        (
            {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []},
            {"farmer": ["PASS"], "hands": [], "market": []},
        ),
    ]
    steps = []
    for step in range(3):
        frame = []
        for seat in range(2):
            frame.append(
                {
                    "observation": _observation(
                        seat, step, 100 - step * 10, include_step=seat == 0
                    ),
                    "action": actions[step][seat],
                    "reward": 0,
                    "status": "DONE" if step == 2 else "ACTIVE",
                    "info": {},
                }
            )
        steps.append(frame)
    return {
        "id": episode_id,
        "info": {"EpisodeId": episode_id, "TeamNames": list(teams)},
        "configuration": {"turnsPerDay": 24},
        "rewards": [100, 80],
        "statuses": ["DONE", "DONE"],
        "steps": steps,
    }


class ReplayPipelineTests(unittest.TestCase):
    def _write(self, directory, name, raw):
        path = Path(directory) / name
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path

    def test_transition_uses_next_frames_action(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = load_replay(self._write(directory, "one.json", _replay()))
            transitions = list(iter_transitions(replay, "Us"))

        self.assertEqual(len(transitions), 2)
        self.assertEqual(transitions[0].step, 0)
        self.assertEqual(transitions[0].action["farmer"], ["EAST"])
        self.assertEqual(transitions[0].next_observation["step"], 1)
        self.assertNotIn("remainingOverageTime", transitions[0].observation)

    def test_missing_step_is_derived_for_second_seat(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = load_replay(self._write(directory, "one.json", _replay()))
            transitions = list(iter_transitions(replay, "Them"))

        self.assertEqual([transition.step for transition in transitions], [0, 1])
        self.assertEqual(transitions[0].action["farmer"], ["WEST"])

    def test_self_play_includes_both_matching_seats(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = load_replay(
                self._write(directory, "self.json", _replay(teams=("Us", "Us")))
            )
            transitions = list(iter_transitions(replay, "Us"))

        self.assertEqual(len(transitions), 4)
        self.assertEqual({item.seat for item in transitions}, {0, 1})
        self.assertTrue(all(item.episode_type == "validation" for item in transitions))

    def test_team_inference_and_dataset_files(self):
        with tempfile.TemporaryDirectory() as directory:
            replay_dir = Path(directory) / "replays"
            output_dir = Path(directory) / "dataset"
            replay_dir.mkdir()
            paths = [
                self._write(replay_dir, "one.json", _replay(123, ("Us", "A"))),
                self._write(replay_dir, "two.json", _replay(124, ("B", "Us"))),
            ]
            self.assertEqual(infer_team_name(paths), "Us")
            manifest = build_dataset(replay_dir, output_dir)

            with gzip.open(output_dir / "transitions.jsonl.gz", "rt") as source:
                records = [json.loads(line) for line in source]
            with (output_dir / "episodes.csv").open(newline="") as source:
                rows = list(csv.DictReader(source))

        self.assertEqual(manifest["unique_episodes_used"], 2)
        self.assertEqual(manifest["trajectories"], 2)
        self.assertEqual(manifest["transitions"], 4)
        self.assertEqual(len(records), 4)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["unused_seed_cost"], "20")

    def test_summary_counts_purchased_seed_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = load_replay(self._write(directory, "one.json", _replay()))
            summary = summarize_trajectory(replay, 0)

        self.assertEqual(summary["bought_seed_units"], 2)
        self.assertEqual(summary["bought_seed_cost"], 20)
        self.assertEqual(summary["unused_seed_units"], 2)

    def test_summary_counts_lost_starting_animal(self):
        raw = _replay()
        raw["steps"][0][0]["observation"]["private"]["shed"]["COW"] = 1
        with tempfile.TemporaryDirectory() as directory:
            replay = load_replay(self._write(directory, "one.json", raw))
            summary = summarize_trajectory(replay, 0)

        self.assertEqual(summary["estimated_lost_animals"], 1)

    def test_teacher_dataset_extracts_both_foreign_players(self):
        with tempfile.TemporaryDirectory() as directory:
            replay_dir = Path(directory) / "teacher_replays"
            output_dir = Path(directory) / "teacher_processed"
            replay_dir.mkdir()
            self._write(replay_dir, "top-game.json", _replay(500, ("Top A", "Top B")))

            manifest = build_teacher_transitions(
                replay_dir,
                output_dir,
                minimum_steps=2,
            )
            with gzip.open(output_dir / "transitions.jsonl.gz", "rt") as source:
                records = [json.loads(line) for line in source]
            with (output_dir / "episodes.csv").open(newline="") as source:
                episodes = list(csv.DictReader(source))

        self.assertEqual(manifest["selected_trajectories"], 2)
        self.assertEqual(manifest["transitions"], 4)
        self.assertEqual({row["team"] for row in episodes}, {"Top A", "Top B"})
        self.assertEqual({row["seat"] for row in records}, {0, 1})
        self.assertTrue(all(row["source_type"] == "teacher_replay" for row in records))

    def test_teacher_dataset_can_keep_only_the_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            replay_dir = Path(directory) / "teacher_replays"
            output_dir = Path(directory) / "teacher_processed"
            replay_dir.mkdir()
            self._write(replay_dir, "top-game.json", _replay(501, ("Winner", "Loser")))

            manifest = build_teacher_transitions(
                replay_dir,
                output_dir,
                winner_only=True,
                minimum_steps=2,
            )
            with gzip.open(output_dir / "transitions.jsonl.gz", "rt") as source:
                records = [json.loads(line) for line in source]

        self.assertEqual(manifest["selected_trajectories"], 1)
        self.assertEqual(manifest["transitions"], 2)
        self.assertEqual({row["team"] for row in records}, {"Winner"})
        self.assertTrue(all(row["teacher_winner"] for row in records))


if __name__ == "__main__":
    unittest.main()
