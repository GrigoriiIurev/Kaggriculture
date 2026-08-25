import tarfile
import tempfile
import unittest
from pathlib import Path

from package_league_submission import build_league_submission
from src.kaggriculture.data.feature_extractor import FeatureExtractor
from src.kaggriculture.league.evaluator import load_agent_file, materialize_main
from src.kaggriculture.rl.train_league_controller import save_fallback_policy
from src.kaggriculture.rl.search_counterfactual_market import save_rule_policy
from src.kaggriculture.rl.train_meta_controller import (
    save_fallback_policy as save_meta_fallback,
)


class PackageLeagueSubmissionTests(unittest.TestCase):
    def test_builds_complete_gzip_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            league = root / "league.npz"
            incumbent = root / "incumbent.npz"
            expert = root / "expert.py"
            output = root / "submission.tar.gz"
            save_fallback_policy(league)
            save_meta_fallback(incumbent, FeatureExtractor().feature_count)
            expert.write_text("def agent(obs): return {}\n", encoding="utf-8")

            build_league_submission(output, league, incumbent, expert)

            self.assertEqual(output.read_bytes()[:2], b"\x1f\x8b")
            with tarfile.open(output, "r:gz") as archive:
                names = set(archive.getnames())
            self.assertIn("main.py", names)
            self.assertIn("artifacts/league/league_policy.npz", names)
            self.assertIn("artifacts/league/incumbent_meta_policy.npz", names)
            self.assertIn("artifacts/league/expert_agent.py", names)
            main = materialize_main(output, root / "unpacked")
            self.assertTrue(callable(load_agent_file(main)))

    def test_loads_packaged_counterfactual_rule_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            league = root / "league.npz"
            incumbent = root / "incumbent.npz"
            expert = root / "expert.py"
            output = root / "submission.tar.gz"
            configuration = {
                "minimum_price_ratios": [1.2] * 4,
                "minimum_stocks": [5] * 4,
                "sale_choices": [2] * 4,
                "late_days": [27] * 4,
                "demand_bonuses": [0.0] * 4,
                "rising_price_bonuses": [0.0] * 4,
            }
            save_rule_policy(league, configuration)
            save_meta_fallback(incumbent, FeatureExtractor().feature_count)
            expert.write_text("def agent(obs): return {}\n", encoding="utf-8")

            build_league_submission(output, league, incumbent, expert)

            main = materialize_main(output, root / "unpacked")
            self.assertTrue(callable(load_agent_file(main)))


if __name__ == "__main__":
    unittest.main()
