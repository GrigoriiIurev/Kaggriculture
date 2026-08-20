import unittest

from main import agent
from src.kaggriculture.core.state_parser import Position, parse_observation
from src.kaggriculture.planning.task_generator import MAINTENANCE, PRODUCTION, SURVIVAL, Task
from src.kaggriculture.planning.worker_planner import (
    EXECUTE_TASK,
    IDLE,
    MOVE_TO_TASK,
    PICKUP_ITEM,
    WorkerPlanner,
)


def farm(tiles, farmer, hands=None):
    return {
        "money": 3000,
        "tiles": tiles,
        "farmer": list(farmer),
        "hands": [list(position) for position in (hands or [])],
        "unlocked_quadrants": ["NW"],
        "hires_today": len(hands or []),
    }


def make_state(
    *,
    farmer=(0, 0),
    hands=(),
    shed=None,
    inventories=None,
    step=100,
    day=4,
    hour=4,
    board_size=6,
):
    tiles = [[None for _ in range(board_size)] for _ in range(board_size)]
    worker_count = 1 + len(hands)
    raw = {
        "player": 0,
        "step": step,
        "day": day,
        "hour": hour,
        "farms": [
            farm(tiles, farmer, hands),
            farm(tiles, (0, 0)),
        ],
        "private": {
            "shed": dict(shed or {}),
            "seeds": {},
            "inventories": (
                [dict(inventory) for inventory in inventories]
                if inventories is not None
                else [{} for _ in range(worker_count)]
            ),
        },
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }
    return parse_observation(raw)


def task(
    operation,
    target,
    priority,
    *,
    required_item=None,
    critical=False,
):
    return Task(
        operation=operation,
        target=Position(*target),
        priority=priority,
        category=SURVIVAL if critical else PRODUCTION,
        reason="test_task",
        required_item=required_item,
        critical=critical,
    )


class WorkerPlannerTests(unittest.TestCase):
    def test_executes_task_when_worker_is_on_target(self):
        state = make_state(farmer=(1, 1))
        planner = WorkerPlanner()

        plan = planner.plan(state, [task("WATER", (1, 1), 900)])

        self.assertEqual(plan.action["farmer"], ["WATER"])
        self.assertEqual(plan.decisions[0].phase, EXECUTE_TASK)
        self.assertEqual(plan.decisions[0].waypoint, Position(1, 1))

    def test_moves_horizontally_then_vertically_toward_task(self):
        state = make_state(farmer=(0, 0))
        planner = WorkerPlanner()

        first = planner.plan(state, [task("DIG", (2, 2), 100)])
        state_on_column = make_state(farmer=(2, 0), step=102, hour=6)
        second = planner.plan(state_on_column, [task("DIG", (2, 2), 100)])

        self.assertEqual(first.action["farmer"], ["EAST"])
        self.assertEqual(second.action["farmer"], ["SOUTH"])
        self.assertEqual(first.decisions[0].phase, MOVE_TO_TASK)

    def test_assigns_each_task_to_nearest_available_worker(self):
        state = make_state(farmer=(0, 0), hands=((5, 5),))
        planner = WorkerPlanner()
        tasks = [
            task("FEED", (5, 4), 1000),
            task("WATER", (0, 1), 900),
        ]

        plan = planner.plan(state, tasks)

        self.assertEqual(plan.action["farmer"], ["SOUTH"])
        self.assertEqual(plan.action["hands"], [["NORTH"]])
        self.assertEqual(plan.decisions[0].task.operation, "WATER")
        self.assertEqual(plan.decisions[1].task.operation, "FEED")

    def test_assigns_only_highest_priority_task_on_one_tile(self):
        state = make_state(farmer=(0, 0), hands=((0, 0),))
        planner = WorkerPlanner()
        tasks = [
            task("HARVEST", (0, 0), 650),
            task("WATER", (0, 0), 900),
        ]

        plan = planner.plan(state, tasks)

        self.assertEqual(plan.action["farmer"], ["WATER"])
        self.assertEqual(plan.action["hands"], [["PASS"]])
        self.assertEqual(len(plan.assigned_tasks), 1)

    def test_picks_up_required_item_at_shed_before_travel(self):
        planner = WorkerPlanner()
        feed = task("FEED", (5, 2), 1300, required_item="WHEAT", critical=True)
        at_shed = make_state(
            farmer=(2, 2),
            shed={"WHEAT": 1},
            inventories=[{}],
        )

        pickup_plan = planner.plan(at_shed, [feed])

        self.assertEqual(pickup_plan.action["farmer"], ["PICKUP", "WHEAT", 1])
        self.assertEqual(pickup_plan.decisions[0].phase, PICKUP_ITEM)

        carrying = make_state(
            farmer=(2, 2),
            shed={},
            inventories=[{"WHEAT": 1}],
            step=101,
            hour=5,
        )
        travel_plan = planner.plan(carrying, [feed])

        self.assertEqual(travel_plan.action["farmer"], ["EAST"])
        self.assertEqual(travel_plan.decisions[0].phase, MOVE_TO_TASK)

    def test_skips_item_task_when_no_worker_or_shed_has_item(self):
        state = make_state(farmer=(0, 0), shed={}, inventories=[{}])
        planner = WorkerPlanner()
        tasks = [
            task("FEED", (5, 5), 1300, required_item="WHEAT", critical=True),
            Task(
                operation="DIG",
                target=Position(1, 0),
                priority=100,
                category=MAINTENANCE,
                reason="clear_weed",
            ),
        ]

        plan = planner.plan(state, tasks)

        self.assertEqual(plan.action["farmer"], ["EAST"])
        self.assertEqual(plan.decisions[0].task.operation, "DIG")

    def test_does_not_reserve_one_shed_item_for_two_workers(self):
        state = make_state(
            farmer=(2, 2),
            hands=((3, 2),),
            shed={"WHEAT": 1},
            inventories=[{}, {}],
        )
        planner = WorkerPlanner()
        tasks = [
            task("FEED", (0, 0), 1300, required_item="WHEAT", critical=True),
            task("FEED", (5, 5), 1290, required_item="WHEAT", critical=True),
        ]

        plan = planner.plan(state, tasks)

        assigned = [decision for decision in plan.decisions if decision.task is not None]
        idle = [decision for decision in plan.decisions if decision.phase == IDLE]
        self.assertEqual(len(assigned), 1)
        self.assertEqual(len(idle), 1)
        self.assertEqual(assigned[0].phase, PICKUP_ITEM)

    def test_continuity_keeps_a_worker_on_a_recent_route(self):
        planner = WorkerPlanner(continuity_bonus=3)
        tasks = [
            task("DIG", (0, 2), 100),
            task("DIG", (4, 2), 100),
        ]
        first_state = make_state(farmer=(1, 2), hands=((3, 2),))

        first_plan = planner.plan(first_state, tasks)
        self.assertEqual(first_plan.decisions[0].task.target, Position(0, 2))
        self.assertEqual(first_plan.decisions[1].task.target, Position(4, 2))

        crossed_state = make_state(
            farmer=(3, 2),
            hands=((1, 2),),
            step=101,
            hour=5,
        )
        second_plan = planner.plan(crossed_state, tasks)

        self.assertEqual(second_plan.decisions[0].task.target, Position(0, 2))
        self.assertEqual(second_plan.decisions[1].task.target, Position(4, 2))

    def test_higher_priority_task_preempts_previous_assignment(self):
        planner = WorkerPlanner(continuity_bonus=100)
        state = make_state(farmer=(0, 0))
        old_task = task("DIG", (5, 5), 100)
        planner.plan(state, [old_task])

        next_state = make_state(farmer=(1, 0), step=101, hour=5)
        emergency = task("WATER", (0, 1), 1300, critical=True)
        plan = planner.plan(next_state, [old_task, emergency])

        self.assertEqual(plan.decisions[0].task, emergency)

    def test_ignores_task_outside_board(self):
        state = make_state(farmer=(0, 0))
        planner = WorkerPlanner()

        plan = planner.plan(state, [task("DIG", (10, 10), 100)])

        self.assertEqual(plan.action["farmer"], ["PASS"])
        self.assertEqual(plan.decisions[0].phase, IDLE)

    def test_main_agent_returns_worker_and_market_commands(self):
        state = make_state(farmer=(0, 0), hands=((1, 0),))
        raw = {
            "player": state.player,
            "step": state.step,
            "day": state.day,
            "hour": state.hour,
            "farms": [
                {
                    "money": farm_state.money,
                    "tiles": [
                        [
                            None if tile.is_empty else "LOCKED"
                            for tile in row
                        ]
                        for row in farm_state.tiles
                    ],
                    "farmer": [farm_state.farmer.x, farm_state.farmer.y],
                    "hands": [[hand.x, hand.y] for hand in farm_state.hands],
                    "unlocked_quadrants": list(farm_state.unlocked_quadrants),
                    "hires_today": farm_state.hires_today,
                }
                for farm_state in state.farms
            ],
            "private": {
                "shed": {},
                "seeds": {},
                "inventories": [{}, {}],
            },
            "market": {"inventory": {}, "prices": {}},
            "town": {"unlocked_shops": []},
        }

        action = agent(raw)

        self.assertEqual(set(action), {"farmer", "hands", "market"})
        self.assertIsInstance(action["farmer"], list)
        self.assertEqual(len(action["hands"]), 1)
        self.assertLessEqual(len(action["market"]), 10)
        self.assertIn(["HIRE"], action["market"])


if __name__ == "__main__":
    unittest.main()
