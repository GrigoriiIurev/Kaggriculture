import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from src.kaggriculture.analysis.replay_warehouse import build_replay_warehouse
from tests.test_replay_pipeline import _replay


class ReplayWarehouseTests(unittest.TestCase):
    def test_builds_compact_episode_daily_and_market_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replays = root / "replays"
            output = root / "analysis"
            replays.mkdir()
            (replays / "episode-123-replay.json").write_text(
                json.dumps(_replay()), encoding="utf-8"
            )
            index = root / "episode_index.json"
            index.write_text(
                json.dumps(
                    {
                        "submission": {"ref": 999},
                        "episodes": [
                            {"id": 123, "createTime": "2026-08-24T00:00:00"}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = build_replay_warehouse(
                replays,
                output,
                team_name="Us",
                episode_index_path=index,
            )
            with (output / "episodes.csv").open(newline="") as stream:
                episodes = list(csv.DictReader(stream))
            with gzip.open(output / "daily_macro.jsonl.gz", "rt") as stream:
                daily = [json.loads(line) for line in stream]
            with gzip.open(output / "market_decisions.jsonl.gz", "rt") as stream:
                market = [json.loads(line) for line in stream]

        self.assertEqual(manifest["submission_id"], 999)
        self.assertEqual(manifest["rows"]["episodes"], 1)
        self.assertEqual(manifest["rows"]["trajectory_profiles"], 2)
        self.assertEqual(manifest["rows"]["daily_macro"], 2)
        self.assertEqual(manifest["rows"]["market_decisions"], 4)
        self.assertEqual(episodes[0]["created_time"], "2026-08-24T00:00:00")
        self.assertEqual(episodes[0]["result"], "win")
        winner_rows = [row for row in market if row["team"] == "Us"]
        self.assertTrue(all(row["is_winner"] for row in winner_rows))
        self.assertEqual(
            winner_rows[0]["market_orders"], [["BUY_SEED", "WHEAT", 2]]
        )
        us_day = next(row for row in daily if row["team"] == "Us")
        self.assertEqual(us_day["market_units"]["BUY_SEED:WHEAT"], 2)


if __name__ == "__main__":
    unittest.main()
