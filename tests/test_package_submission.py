import tarfile
import tempfile
import unittest
from pathlib import Path

from package_submission import WORKER_MODEL_ARCHIVE_PATH, build_submission


class PackageSubmissionTests(unittest.TestCase):
    def test_embeds_worker_model_and_inference_code(self):
        with tempfile.TemporaryDirectory() as directory:
            worker_model = Path(directory) / "worker.npz"
            worker_model.write_bytes(b"test model")
            output = Path(directory) / "submission.tar.gz"

            result = build_submission(output, worker_model)
            magic = output.read_bytes()[:2]
            with tarfile.open(output, "r:gz") as archive:
                names = set(archive.getnames())

        self.assertEqual(result["agent"], "behavior_cloning_workers")
        self.assertEqual(magic, b"\x1f\x8b")
        self.assertIn("main.py", names)
        self.assertIn(WORKER_MODEL_ARCHIVE_PATH, names)
        self.assertIn(
            "src/kaggriculture/learning/behavior_model.py",
            names,
        )
        self.assertIn(
            "src/kaggriculture/data/worker_dataset.py",
            names,
        )

    def test_rejects_missing_worker_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                build_submission(
                    Path(directory) / "submission.tar.gz",
                    Path(directory) / "missing.npz",
                )


if __name__ == "__main__":
    unittest.main()
