import base64
import unittest
import zlib

from src.kaggriculture.league.notebook_source import extract_agent_source


AGENT_A = b"def agent(obs):\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
AGENT_B = b"def agent(obs, config=None):\n    return {'farmer': ['NORTH'], 'hands': [], 'market': []}\n"


class LeagueNotebookSourceTests(unittest.TestCase):
    def test_extracts_writefile_cell_with_string_source(self):
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": "%%writefile main.py\n" + AGENT_A.decode(),
                }
            ]
        }
        payload, details = extract_agent_source(notebook)
        self.assertEqual(payload, AGENT_A)
        self.assertEqual(details["method"], "writefile")

    def test_extracts_base85_zlib_payload_without_executing_cell(self):
        encoded = base64.b85encode(zlib.compress(AGENT_A)).decode("ascii")
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": f"payload = {encoded!r}\nraise RuntimeError('must not run')\n",
                }
            ]
        }
        payload, details = extract_agent_source(notebook)
        self.assertEqual(payload, AGENT_A)
        self.assertEqual(details["method"], "payload")

    def test_chooses_last_packaged_artifact(self):
        notebook = {
            "cells": [
                {"cell_type": "code", "source": f"AGENT_SOURCE = {AGENT_A.decode()!r}"},
                {"cell_type": "code", "source": f"AGENT_SOURCE = {AGENT_B.decode()!r}"},
            ]
        }
        payload, details = extract_agent_source(notebook)
        self.assertEqual(payload, AGENT_B)
        self.assertEqual(details["cell_index"], 1)


if __name__ == "__main__":
    unittest.main()
