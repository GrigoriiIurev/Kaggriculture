import unittest
from types import SimpleNamespace

from src.kaggriculture.core.state_parser import (
    COOP,
    EMPTY,
    LOCKED,
    PASTURE,
    PLANT,
    Position,
    StateParseError,
    parse_observation,
)


def plant(crop="WHEAT", watered=False, yield_units=2):
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": 1,
        "watered_today": watered,
        "consecutive_unwatered": 1,
        "yield_units": yield_units,
        "max_lifespan_step": 120,
        "fertilized_until_day": -1,
    }


def animal(kind="PASTURE", animal_type="COW", fed=False):
    return {
        "kind": kind,
        "animal": animal_type,
        "placed_day": 1,
        "yield_units": 1,
        "fed_today": fed,
        "consecutive_unfed": 0,
        "cared_today": False,
        "fertilizer_available": True,
        "pending_care_bonus": 2,
    }


def farm(money, tiles, hands=None):
    return {
        "money": money,
        "tiles": tiles,
        "farmer": [0, 0],
        "hands": hands or [],
        "unlocked_quadrants": ["NW"],
        "hires_today": len(hands or []),
    }


def observation():
    my_tiles = [
        [plant(), animal(), None],
        [{"kind": "WEED"}, "LOCKED", animal(kind="COOP", animal_type=None)],
    ]
    opponent_tiles = [
        [None, None, None],
        [None, "LOCKED", None],
    ]
    return {
        "player": 0,
        "step": 50,
        "day": 2,
        "hour": 2,
        "farms": [farm(1200, my_tiles, hands=[[1, 0]]), farm(900, opponent_tiles)],
        "private": {
            "shed": {"WHEAT": 20, "MILK": 3},
            "seeds": {"WHEAT": 4},
            "inventories": [{"WHEAT": 1}, {}],
        },
        "market": {
            "inventory": {"WHEAT": 9980, "MILK": 10010},
            "prices": {"WHEAT": 27, "MILK": 140},
        },
        "town": {
            "unlocked_shops": ["BAKERY", "BAKERY", "YARN_STORE"],
        },
    }


class StateParserTests(unittest.TestCase):
    def test_parses_complete_observation(self):
        state = parse_observation(observation())

        self.assertEqual(state.player, 0)
        self.assertEqual(state.step, 50)
        self.assertEqual(state.me.money, 1200.0)
        self.assertEqual(state.opponent.money, 900.0)
        self.assertEqual(state.me.width, 3)
        self.assertEqual(state.me.height, 2)
        self.assertEqual(state.market.price("MILK"), 140)
        self.assertEqual(state.town.shop_counts, {"BAKERY": 2, "YARN_STORE": 1})

    def test_builds_useful_tile_collections(self):
        state = parse_observation(observation())

        self.assertEqual(len(state.me.plants), 1)
        self.assertEqual(state.me.plants[0].kind, PLANT)
        self.assertTrue(state.me.plants[0].needs_water)
        self.assertEqual(len(state.me.animals), 1)
        self.assertEqual(state.me.animals[0].kind, PASTURE)
        self.assertTrue(state.me.animals[0].needs_feed)
        self.assertEqual(len(state.me.weeds), 1)
        self.assertEqual(len(state.me.empty_tiles), 1)
        self.assertEqual(state.me.tile_at(Position(1, 1)).kind, LOCKED)
        self.assertEqual(state.me.tile_at(Position(2, 0)).kind, EMPTY)
        self.assertEqual(state.me.tile_at(Position(2, 1)).kind, COOP)

    def test_combines_worker_positions_with_private_inventories(self):
        state = parse_observation(observation())

        self.assertEqual(len(state.units), 2)
        self.assertTrue(state.units[0].is_farmer)
        self.assertEqual(state.units[0].inventory, {"WHEAT": 1})
        self.assertFalse(state.units[1].is_farmer)
        self.assertEqual(state.units[1].position, Position(1, 0))

    def test_calculates_capacity_and_time(self):
        state = parse_observation(
            observation(),
            {"turnsPerDay": 24, "episodeSteps": 100, "shedCapacity": 30},
        )

        self.assertEqual(state.private.shed_used, 23)
        self.assertEqual(state.private.shed_free, 7)
        self.assertEqual(state.remaining_steps, 50)
        self.assertFalse(state.is_last_step)

    def test_derives_step_when_framework_does_not_supply_it(self):
        raw = observation()
        del raw["step"]

        state = parse_observation(raw)

        self.assertEqual(state.step, 50)

    def test_accepts_attribute_based_kaggle_objects(self):
        raw = observation()
        raw["town"] = SimpleNamespace(
            unlocked_shops=raw["town"]["unlocked_shops"]
        )
        configuration = SimpleNamespace(
            turnsPerDay=24,
            episodeSteps=720,
            shedCapacity=100,
        )

        state = parse_observation(SimpleNamespace(**raw), configuration)

        self.assertEqual(state.town.count("BAKERY"), 2)

    def test_reports_inventory_count_mismatch(self):
        raw = observation()
        raw["private"]["inventories"] = [{}]

        with self.assertRaisesRegex(StateParseError, "expected 2, got 1"):
            parse_observation(raw)

    def test_reports_bad_tile_with_location(self):
        raw = observation()
        raw["farms"][0]["tiles"][1][2] = "STONE"

        with self.assertRaisesRegex(StateParseError, r"tiles\[1\]\[2\]"):
            parse_observation(raw)


if __name__ == "__main__":
    unittest.main()
