import unittest

from scripts.sync_submission_replays import select_latest_completed_submission


class SyncSubmissionReplaysTests(unittest.TestCase):
    def test_selects_newest_completed_submission(self):
        submissions = [
            {
                "ref": 1,
                "date": "2026-08-20T00:00:00",
                "status": "SubmissionStatus.COMPLETE",
            },
            {
                "ref": 2,
                "date": "2026-08-21T00:00:00",
                "status": "SubmissionStatus.PENDING",
            },
            {
                "ref": 3,
                "date": "2026-08-22T00:00:00",
                "status": "SubmissionStatus.COMPLETE",
            },
        ]

        selected = select_latest_completed_submission(submissions)

        self.assertEqual(selected["ref"], 3)

    def test_rejects_empty_completed_set(self):
        with self.assertRaisesRegex(RuntimeError, "No completed"):
            select_latest_completed_submission([])


if __name__ == "__main__":
    unittest.main()
