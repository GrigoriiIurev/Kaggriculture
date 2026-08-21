import tarfile
import tempfile
import unittest
from pathlib import Path

from package_rl_submission import build_rl_submission
from src.kaggriculture.data.feature_extractor import FeatureExtractor
from src.kaggriculture.rl.train_meta_controller import save_fallback_policy


class PackageRlSubmissionTests(unittest.TestCase):
    def test_builds_gzip_archive_with_model_and_expert(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.npz"
            expert = root / "expert.py"
            output = root / "submission.tar.gz"
            save_fallback_policy(model, FeatureExtractor().feature_count)
            expert.write_text("def agent(obs): return {}\n", encoding="utf-8")

            build_rl_submission(output, model, expert)

            self.assertEqual(output.read_bytes()[:2], b"\x1f\x8b")
            with tarfile.open(output, "r:gz") as archive:
                names = set(archive.getnames())
            self.assertIn("main.py", names)
            self.assertIn("artifacts/rl/meta_policy.npz", names)
            self.assertIn("artifacts/rl/expert_agent.py", names)


if __name__ == "__main__":
    unittest.main()
