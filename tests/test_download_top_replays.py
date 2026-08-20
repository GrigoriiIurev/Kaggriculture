import unittest

from scripts.download_top_replays import (
    _best_submission,
    _completed_public_episode_ids,
    _next_page_token,
    _parse_json_list,
)


class DownloadTopReplaysTests(unittest.TestCase):
    def test_parses_leaderboard_after_page_token(self):
        output = 'Next Page Token = abc123\n[{"teamId": 7}]\n'

        self.assertEqual(_parse_json_list(output, "leaderboard"), [{"teamId": 7}])
        self.assertEqual(_next_page_token(output), "abc123")

    def test_selects_highest_scoring_public_submission(self):
        submissions = [
            {"id": 10, "publicScore": "2500.0", "dateSubmitted": "2026-01-02"},
            {"id": 11, "publicScore": "3100.5", "dateSubmitted": "2026-01-01"},
        ]

        self.assertEqual(_best_submission(submissions)["id"], 11)

    def test_keeps_newest_completed_public_episodes(self):
        episodes = [
            {
                "id": 1,
                "createTime": "2026-01-01T00:00:00",
                "state": "EpisodeState.COMPLETED",
                "type": "EpisodeType.EPISODE_TYPE_PUBLIC",
            },
            {
                "id": 2,
                "createTime": "2026-01-03T00:00:00",
                "state": "EpisodeState.COMPLETED",
                "type": "EpisodeType.EPISODE_TYPE_PUBLIC",
            },
            {
                "id": 3,
                "createTime": "2026-01-04T00:00:00",
                "state": "EpisodeState.RUNNING",
                "type": "EpisodeType.EPISODE_TYPE_PUBLIC",
            },
            {
                "id": 4,
                "createTime": "2026-01-05T00:00:00",
                "state": "EpisodeState.COMPLETED",
                "type": "EpisodeType.EPISODE_TYPE_VALIDATION",
            },
        ]

        self.assertEqual(_completed_public_episode_ids(episodes, limit=1), [2])


if __name__ == "__main__":
    unittest.main()
