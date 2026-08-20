import unittest

from src.kaggriculture.core.action_codec import (
    ARGUMENT_TO_ID,
    MARKET_OPERATION_TO_ID,
    WORKER_OPERATION_TO_ID,
    ActionDecoder,
    ActionEncoder,
    ActionEncodingError,
    EncodedCommand,
)


class ActionCodecTests(unittest.TestCase):
    def setUp(self):
        self.encoder = ActionEncoder()
        self.decoder = ActionDecoder()

    def test_round_trips_worker_commands(self):
        commands = [
            ["PASS"],
            ["NORTH"],
            ["WATER"],
            ["BUILD_PASTURE"],
            ["PLANT", "WHEAT"],
            ["PICKUP", "WHEAT", 4],
            ["PLACE", "COW", 1],
        ]
        for command in commands:
            with self.subTest(command=command):
                encoded = self.encoder.encode_worker(command)
                decoded = self.decoder.decode_worker(
                    encoded.operation_id, encoded.argument_id, encoded.quantity
                )
                self.assertEqual(decoded, command)

    def test_round_trips_market_commands(self):
        commands = [
            ["HIRE"],
            ["BUY_LAND"],
            ["BUY_SEED", "MELON", 3],
            ["BUY_PRODUCT", "FERTILIZER", 2],
            ["BUY_ANIMAL", "SHEEP", 1],
            ["SELL", "MILK", 12],
        ]
        for command in commands:
            with self.subTest(command=command):
                encoded = self.encoder.encode_market(command)
                decoded = self.decoder.decode_market(
                    encoded.operation_id, encoded.argument_id, encoded.quantity
                )
                self.assertEqual(decoded, command)

    def test_encodes_and_decodes_complete_action(self):
        raw = {
            "farmer": ["PLANT", "TOMATO"],
            "hands": [["WEST"], ["FEED"]],
            "market": [["BUY_SEED", "WHEAT", 2], ["HIRE"]],
        }

        encoded = self.encoder.encode_action(raw, expected_hands=2)
        decoded = self.decoder.decode_action(
            encoded.farmer, encoded.hands, encoded.market
        )

        self.assertEqual(decoded, raw)

    def test_encoder_rejects_impossible_argument_combinations(self):
        with self.assertRaises(ActionEncodingError):
            self.encoder.encode_worker(["PLANT", "COW"])
        with self.assertRaises(ActionEncodingError):
            self.encoder.encode_market(["BUY_PRODUCT", "MILK", 2])
        with self.assertRaisesRegex(ActionEncodingError, "Expected 2"):
            self.encoder.encode_action(
                {"farmer": ["PASS"], "hands": [], "market": []},
                expected_hands=2,
            )

    def test_decoder_falls_back_for_invalid_predictions(self):
        self.assertEqual(self.decoder.decode_worker(999), ["PASS"])
        self.assertEqual(
            self.decoder.decode_worker(
                WORKER_OPERATION_TO_ID["PLANT"], ARGUMENT_TO_ID["COW"]
            ),
            ["PASS"],
        )
        self.assertIsNone(
            self.decoder.decode_market(
                MARKET_OPERATION_TO_ID["BUY_PRODUCT"], ARGUMENT_TO_ID["MILK"]
            )
        )

    def test_decoder_rounds_and_clamps_quantity(self):
        command = EncodedCommand(
            WORKER_OPERATION_TO_ID["PICKUP"], ARGUMENT_TO_ID["WHEAT"], 200
        )

        decoded = self.decoder.decode_worker(
            command.operation_id, command.argument_id, command.quantity
        )

        self.assertEqual(decoded, ["PICKUP", "WHEAT", 100])


if __name__ == "__main__":
    unittest.main()
