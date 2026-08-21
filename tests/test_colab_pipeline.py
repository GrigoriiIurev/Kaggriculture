import tempfile
import unittest
from pathlib import Path

from package_submission import build_submission
from run_colab_pipeline import (
    copy_submission_to_drive,
    restore_checkpoint,
    save_checkpoint,
    submission_fingerprint,
)


class ColabPipelineTests(unittest.TestCase):
    def test_submission_fingerprint_ignores_archive_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "worker.npz"
            model.write_bytes(b"same model")
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"

            build_submission(first, model)
            build_submission(second, model)

            self.assertEqual(
                submission_fingerprint(first),
                submission_fingerprint(second),
            )

    def test_drive_copy_remains_a_gzip_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "worker.npz"
            model.write_bytes(b"model")
            local = root / "local.tar.gz"
            drive = root / "drive" / "submission.tar.gz"
            drive.parent.mkdir()
            build_submission(local, model)

            copy_submission_to_drive(local, drive)

            self.assertEqual(drive.read_bytes()[:2], b"\x1f\x8b")
            self.assertEqual(local.read_bytes(), drive.read_bytes())

    def test_checkpoint_round_trip_preserves_directory_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "teacher_replays"
            source.mkdir(parents=True)
            (source / "episode.json").write_text(
                '{"id": 123}\n', encoding="utf-8"
            )
            archive = root / "drive" / "replays.tar.gz"

            save_checkpoint(source, archive, temporary_directory=root)
            restored = root / "restored" / "teacher_replays"
            was_restored = restore_checkpoint(archive, restored)

            self.assertTrue(was_restored)
            self.assertEqual(
                (restored / "episode.json").read_text(encoding="utf-8"),
                '{"id": 123}\n',
            )


if __name__ == "__main__":
    unittest.main()
