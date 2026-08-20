import unittest

from src.kaggriculture.planning.economic_planner import BOOTSTRAP, LIQUIDATE, EconomicPlanner
from src.kaggriculture.core.game_data import BASE_PRICES
from src.kaggriculture.core.state_parser import Position, parse_observation


def empty_board():
    return [
        [None if x < 5 and y < 5 else "LOCKED" for x in range(10)]
        for y in range(10)
    ]


def farm(tiles, *, money=3000, hands=(), day=0):
    return {
        "money": money,
        "tiles": tiles,
        "farmer": [4, 4],
        "hands": [list(position) for position in hands],
        "unlocked_quadrants": ["NW"],
        "hires_today": len(hands),
    }


def raw_state(
    *,
    day=0,
    hour=0,
    step=None,
    money=3000,
    shed=None,
    seeds=None,
    inventories=None,
    hands=(),
    tiles=None,
):
    my_tiles = tiles or empty_board()
    opponent_tiles = empty_board()
    worker_count = 1 + len(hands)
    return {
        "player": 0,
        "step": day * 24 + hour if step is None else step,
        "day": day,
        "hour": hour,
        "farms": [
            farm(my_tiles, money=money, hands=hands, day=day),
            farm(opponent_tiles, day=day),
        ],
        "private": {
            "shed": dict(shed or {}),
            "seeds": dict(seeds or {}),
            "inventories": (
                [dict(value) for value in inventories]
                if inventories is not None
                else [{} for _ in range(worker_count)]
            ),
        },
        "market": {
            "inventory": {item: 10000 for item in BASE_PRICES},
            "prices": dict(BASE_PRICES),
        },
        "town": {"unlocked_shops": []},
    }


def cow_tile():
    return {
        "kind": "PASTURE",
        "animal": "COW",
        "placed_day": 0,
        "yield_units": 0,
        "fed_today": False,
        "consecutive_unfed": 0,
        "cared_today": False,
        "fertilizer_available": False,
        "pending_care_bonus": 0,
    }


class EconomicPlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = EconomicPlanner()

    def test_bootstrap_builds_capacity_and_starts_a_budgeted_opening(self):
        plan = self.planner.plan(parse_observation(raw_state()))

        self.assertEqual(plan.phase, BOOTSTRAP)
        self.assertEqual(plan.target_animals, {"COW": 1, "SHEEP": 4})
        self.assertEqual(
            sum(task.operation == "BUILD_PASTURE" for task in plan.tasks),
            5,
        )
        self.assertFalse(any(task.operation in {"PLANT", "PLACE"} for task in plan.tasks))
        self.assertEqual(sum(order[0] == "HIRE" for order in plan.market_orders), 4)
        self.assertLessEqual(len(plan.market_orders), 10)
        self.assertGreaterEqual(
            sum(order[2] for order in plan.market_orders if order[0] == "BUY_ANIMAL"),
            1,
        )

    def test_field_commitments_only_use_items_already_owned(self):
        tiles = empty_board()
        tiles[4][4] = {"kind": "PASTURE"}
        state = parse_observation(
            raw_state(
                tiles=tiles,
                shed={"COW": 1},
                seeds={"WHEAT": 2},
            )
        )

        plan = self.planner.plan(state)
        plant_tasks = [task for task in plan.tasks if task.operation == "PLANT"]
        place_tasks = [task for task in plan.tasks if task.operation == "PLACE"]

        self.assertLessEqual(len(plant_tasks), 2)
        self.assertEqual(len(place_tasks), 1)
        self.assertEqual(place_tasks[0].required_item, "COW")

    def test_keeps_wheat_needed_for_animals_while_selling_produce(self):
        tiles = empty_board()
        tiles[4][4] = cow_tile()
        plan = self.planner.plan(
            parse_observation(
                raw_state(tiles=tiles, shed={"WHEAT": 5, "MELON": 10})
            )
        )

        sells = {order[1]: order[2] for order in plan.market_orders if order[0] == "SELL"}
        self.assertEqual(plan.feed_reserve, 6)
        self.assertNotIn("WHEAT", sells)
        self.assertGreater(sells.get("MELON", 0), 0)

    def test_cash_reserve_does_not_block_emergency_feed(self):
        tiles = empty_board()
        tiles[4][4] = cow_tile()
        plan = self.planner.plan(
            parse_observation(raw_state(tiles=tiles, money=100, shed={}))
        )

        feed_order = next(
            order for order in plan.market_orders if order[:2] == ("BUY_PRODUCT", "WHEAT")
        )
        self.assertEqual(feed_order, ("BUY_PRODUCT", "WHEAT", 4))

    def test_liquidation_sells_everything_and_stops_investing(self):
        plan = self.planner.plan(
            parse_observation(
                raw_state(
                    day=28,
                    hour=8,
                    step=680,
                    shed={"WHEAT": 7, "MELON": 4, "FERTILIZER": 2},
                    inventories=[{"MILK": 3}],
                )
            )
        )

        self.assertEqual(plan.phase, LIQUIDATE)
        self.assertTrue(plan.market_orders)
        self.assertTrue(all(order[0] == "SELL" for order in plan.market_orders))
        self.assertEqual(
            {order[1]: order[2] for order in plan.market_orders},
            {"MELON": 4, "WHEAT": 7, "FERTILIZER": 2},
        )
        drop = next(task for task in plan.tasks if task.operation == "DROP")
        self.assertEqual(drop.assigned_unit, 0)

    def test_buys_first_land_only_after_the_day_gate(self):
        early = self.planner.plan(parse_observation(raw_state(day=4, money=10000)))
        ready = self.planner.plan(parse_observation(raw_state(day=5, money=10000)))

        self.assertFalse(any(order[0] == "BUY_LAND" for order in early.market_orders))
        self.assertTrue(any(order[0] == "BUY_LAND" for order in ready.market_orders))


if __name__ == "__main__":
    unittest.main()
