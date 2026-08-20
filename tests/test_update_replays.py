import tempfile
import unittest
from pathlib import Path

from scripts.update_replays import (
    _completed_episode_ids,
    _existing_episode_ids,
    _parse_episode_output,
)


class UpdateReplaysTests(unittest.TestCase):
    def test_parses_json_with_cli_hint_after_it(self):
        output = (
            '[{"id": 10, "state": "EpisodeState.COMPLETED"}]\n\n'
            'Use "kaggle competitions replay <episode_id>" to download.\n'
        )

        episodes = _parse_episode_output(output)

        self.assertEqual(_completed_episode_ids(episodes), [10])

    def test_ignores_incomplete_and_invalid_episodes(self):
        episodes = [
            {"id": 12, "state": "EpisodeState.COMPLETED"},
            {"id": 11, "state": "EpisodeState.COMPLETED"},
            {"id": 13, "state": "EpisodeState.RUNNING"},
            {"state": "EpisodeState.COMPLETED"},
        ]

        self.assertEqual(_completed_episode_ids(episodes), [11, 12])

    def test_reads_ids_from_replay_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "episode-12345678-replay.json").touch()
            (root / "87654321.json").touch()
            (root / "notes.json").touch()

            ids = _existing_episode_ids(root)

        self.assertEqual(ids, {12345678, 87654321})


if __name__ == "__main__":
    unittest.main()
