import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from src.kaggriculture.analysis.loss_replay_analyzer import (
    build_loss_replay_analysis,
)


def _farm(money, animals=0, plants=0, hands=1, land=1, **extra):
    return {
        "money": money,
        "animal_total": animals,
        "plant_total": plants,
        "hands": hands,
        "land": land,
        "weeds": extra.get("weeds", 0),
        "endangered_plants": extra.get("endangered_plants", 0),
        "endangered_animals": extra.get("endangered_animals", 0),
    }


def _day(episode, result, day, ours, opponent, opponent_name="Rival"):
    return {
        "episode_id": episode,
        "episode_type": "public",
        "seat": 0,
        "team": "Us",
        "opponent": opponent_name,
        "day": day,
        "result": result,
        "start": {
            "me": _farm(ours["money"] - 100, ours["animals"], ours["plants"]),
            "opponent": _farm(
                opponent["money"] - 100,
                opponent["animals"],
                opponent["plants"],
            ),
        },
        "end": {
            "me": _farm(
                ours["money"], ours["animals"], ours["plants"], **ours.get("extra", {})
            ),
            "opponent": _farm(
                opponent["money"], opponent["animals"], opponent["plants"]
            ),
            "private": {"shed": ours.get("shed", {}), "carried": {}},
        },
        "money_delta": 100,
        "field_actions": {"PASS": 2, "WATER": 8},
        "market_operations": {"SELL": 1},
        "market_units": {},
    }


class LossReplayAnalyzerTests(unittest.TestCase):
    def test_finds_scale_deficit_and_writes_reports(self):
        records = [
            _day(
                1,
                "loss",
                0,
                {"money": 3000, "animals": 0, "plants": 2},
                {"money": 3000, "animals": 0, "plants": 2},
            ),
            _day(
                1,
                "loss",
                5,
                {"money": 4000, "animals": 1, "plants": 3},
                {"money": 5600, "animals": 5, "plants": 4},
            ),
            _day(
                1,
                "loss",
                10,
                {"money": 5000, "animals": 2, "plants": 4},
                {"money": 8000, "animals": 7, "plants": 5},
            ),
            _day(
                2,
                "win",
                10,
                {"money": 9000, "animals": 6, "plants": 6},
                {"money": 7000, "animals": 4, "plants": 5},
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "daily_macro.jsonl.gz"
            output = root / "analysis"
            with gzip.open(source, "wt", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")

            report = build_loss_replay_analysis(source, output, "Us")
            with (output / "loss_days.csv").open(newline="") as stream:
                days = list(csv.DictReader(stream))

            self.assertEqual(report["summary"]["games"], 2)
            self.assertEqual(report["summary"]["losses"], 1)
            self.assertEqual(report["losses"][0]["primary_cause"], "production_scale")
            self.assertEqual(report["losses"][0]["first_lasting_deficit_day"], 5)
            self.assertEqual(len(days), 3)
            self.assertIn("win", report["summary"]["cohort_comparison"])
            self.assertIn("loss", report["summary"]["cohort_comparison"])
            self.assertTrue((output / "loss_report.md").is_file())
            self.assertTrue((output / "loss_diagnostics.json").is_file())


if __name__ == "__main__":
    unittest.main()
