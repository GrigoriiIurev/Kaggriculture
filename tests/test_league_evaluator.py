import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.kaggriculture.league.evaluator import (
    AgentSpec,
    bradley_terry,
    evaluate_league,
    load_agent_file,
    materialize_main,
)


AGENT = "def agent(obs):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"


class LeagueEvaluatorTests(unittest.TestCase):
    def test_bradley_terry_ranks_winner_above_loser(self):
        ratings = bradley_terry(
            [
                {
                    "agent_a": "strong",
                    "agent_b": "weak",
                    "wins_a": 8,
                    "wins_b": 2,
                    "ties": 0,
                }
            ]
        )
        self.assertGreater(ratings["strong"], ratings["weak"])

    def test_materializes_root_main_from_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "main.py"
            source.write_text(AGENT, encoding="utf-8")
            archive = root / "submission.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                bundle.add(source, arcname="main.py")
            result = materialize_main(archive, root / "unpacked")
            self.assertEqual(result.read_text(encoding="utf-8"), AGENT)

    def test_multifile_agent_uses_its_own_src_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src/__init__.py").write_text("", encoding="utf-8")
            (root / "src/bundled_value.py").write_text(
                "VALUE = 'from_submission'\n", encoding="utf-8"
            )
            (root / "main.py").write_text(
                "from src.bundled_value import VALUE\n"
                "def agent(obs):\n"
                "    return {'value': VALUE}\n",
                encoding="utf-8",
            )

            original_src = sys.modules.get("src")
            agent = load_agent_file(root / "main.py")

            self.assertEqual(agent({}), {"value": "from_submission"})
            self.assertIs(sys.modules.get("src"), original_src)

    def test_league_reuses_completed_games(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.py"
            second = root / "b.py"
            first.write_text(AGENT, encoding="utf-8")
            second.write_text(AGENT, encoding="utf-8")
            agents = [AgentSpec("a", first), AgentSpec("b", second)]

            def fake_game(a, b, seed, seat_a, episode_steps):
                return {
                    "key": f"{a.slug}|unused|{b.slug}|unused|{seed}|{seat_a}",
                    "agent_a": a.slug,
                    "agent_b": b.slug,
                    "seed": seed,
                    "seat_a": seat_a,
                    "reward_a": 10.0,
                    "reward_b": 5.0,
                    "margin_a": 5.0,
                    "outcome_a": "win",
                    "status_a": "DONE",
                    "status_b": "DONE",
                    "error": "",
                    "seconds": 0.1,
                }

            # Use the real digest-based key so the second call sees the cache.
            from src.kaggriculture.league import evaluator

            def keyed_game(a, b, seed, seat_a, episode_steps):
                record = fake_game(a, b, seed, seat_a, episode_steps)
                record["key"] = evaluator._result_key(
                    a, b, seed, seat_a, episode_steps
                )
                return record

            with mock.patch.object(evaluator, "_play_game", side_effect=keyed_game) as play:
                first_run = evaluate_league(
                    agents, root / "results", seeds=[1], challenger="a"
                )
                second_run = evaluate_league(
                    agents, root / "results", seeds=[1], challenger="a"
                )
            self.assertEqual(play.call_count, 2)
            self.assertEqual(first_run["completed_games"], 2)
            self.assertEqual(second_run["completed_games"], 2)
            self.assertTrue(json.loads((root / "results/league_summary.json").read_text())["promotion_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
