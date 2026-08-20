import unittest

from src.kaggriculture.core.action_codec import ActionEncoder, MARKET_OPERATION_TO_ID
from src.kaggriculture.data.economic_dataset import _encode_market_orders


class EconomicDatasetTests(unittest.TestCase):
    def test_normalizes_orders_the_game_engine_accepts(self):
        orders = _encode_market_orders(
            [
                ["HIRE", "IGNORED"],
                ["BUY_SEED", "WHEAT", "2", "IGNORED"],
                ["BUY_SEED", "WHEAT"],
            ],
            ActionEncoder(),
            line_number=1,
            max_market_orders=10,
        )

        self.assertEqual(len(orders), 2)
        self.assertEqual(orders[0]["operation_id"], MARKET_OPERATION_TO_ID["HIRE"])
        self.assertEqual(orders[1]["quantity"], 2)


if __name__ == "__main__":
    unittest.main()
