import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.fetch_kaggle_opponent import extract_written_file


class FetchKaggleOpponentTests(unittest.TestCase):
    def test_extracts_writefile_cell_and_checks_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = "def agent(obs):\n    return {'farmer': ['PASS']}\n"
            notebook = {
                "metadata": {
                    "v16_rc5": {
                        "agent_sha256": hashlib.sha256(
                            payload.encode("utf-8")
                        ).hexdigest()
                    }
                },
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["%%writefile main.py\n", payload],
                    }
                ],
            }
            source = root / "agent.ipynb"
            output = root / "opponent.py"
            source.write_text(json.dumps(notebook), encoding="utf-8")

            result = extract_written_file(source, output)

            self.assertEqual(output.read_text(encoding="utf-8"), payload)
            self.assertEqual(result["sha256"], notebook["metadata"]["v16_rc5"]["agent_sha256"])

    def test_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "agent.ipynb"
            source.write_text(
                json.dumps(
                    {
                        "metadata": {"v16_rc5": {"agent_sha256": "wrong"}},
                        "cells": [
                            {
                                "cell_type": "code",
                                "source": ["%%writefile main.py\n", "x = 1\n"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "hash does not match"):
                extract_written_file(source, root / "out.py")


if __name__ == "__main__":
    unittest.main()
