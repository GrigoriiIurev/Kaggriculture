import json
import tempfile
import unittest
from pathlib import Path

from src.kaggriculture.agent import RuleBasedAgent, load_economic_config
from src.kaggriculture.planning.economic_planner import EconomicConfig


class AgentConfigTests(unittest.TestCase):
    def test_missing_config_uses_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_economic_config(Path(directory) / "missing.json")
        self.assertEqual(config, EconomicConfig())

    def test_promoted_payload_loads_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"config": {"cash_reserve": 100}}),
                encoding="utf-8",
            )
            config = load_economic_config(path)
        self.assertEqual(config.cash_reserve, 100)
        self.assertIsInstance(RuleBasedAgent(config), RuleBasedAgent)

    def test_unknown_config_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"config": {"not_a_real_setting": 1}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_economic_config(path)


if __name__ == "__main__":
    unittest.main()
