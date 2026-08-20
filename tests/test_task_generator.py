import unittest

from src.kaggriculture.core.state_parser import Position, parse_observation
from src.kaggriculture.planning.task_generator import (
    CARE,
    COLLECT_FERTILIZER,
    DIG,
    FEED,
    HARVEST,
    MAINTENANCE,
    PRODUCTION,
    SURVIVAL,
    WATER,
    TaskGenerator,
)


def plant(
    *,
    crop="WHEAT",
    planted_day=0,
    watered=False,
    missed_days=0,
    yield_units=0,
    max_lifespan_step=500,
):
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": planted_day,
        "watered_today": watered,
        "consecutive_unwatered": missed_days,
        "yield_units": yield_units,
        "max_lifespan_step": max_lifespan_step,
        "fertilized_until_day": -1,
    }


def animal(
    animal_type,
    *,
    fed=False,
    missed_days=0,
    yield_units=0,
    cared=False,
    fertilizer=False,
):
    return {
        "kind": "COOP" if animal_type == "GOOSE" else "PASTURE",
        "animal": animal_type,
        "placed_day": 1,
        "yield_units": yield_units,
        "fed_today": fed,
        "consecutive_unfed": missed_days,
        "cared_today": cared,
        "fertilizer_available": fertilizer,
        "pending_care_bonus": 0,
    }


def farm(tiles):
    return {
        "money": 3000,
        "tiles": tiles,
        "farmer": [0, 0],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }


def observation():
    my_tiles = [
        [
            plant(missed_days=1),
            plant(missed_days=0, yield_units=2),
            plant(watered=True, yield_units=3, max_lifespan_step=90),
            {"kind": "WEED"},
        ],
        [
            animal(
                "COW",
                missed_days=1,
                yield_units=2,
                fertilizer=True,
            ),
            animal("SHEEP", missed_days=0),
            animal(
                "GOOSE",
                fed=True,
                yield_units=1,
                fertilizer=True,
            ),
            {"kind": "PASTURE", "animal": None},
        ],
    ]
    opponent_tiles = [
        [plant(missed_days=1), None, None, None],
        [None, None, None, animal("COW", missed_days=1)],
    ]
    return {
        "player": 0,
        "step": 100,
        "day": 4,
        "hour": 4,
        "farms": [farm(my_tiles), farm(opponent_tiles)],
        "private": {
            "shed": {"WHEAT": 5},
            "seeds": {},
            "inventories": [{}],
        },
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }


def tasks_at(tasks, position):
    return tuple(task for task in tasks if task.target == position)


class TaskGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.state = parse_observation(observation())
        self.tasks = TaskGenerator().generate(self.state)

    def test_generates_tasks_only_for_our_farm(self):
        self.assertEqual(len(self.tasks), 14)
        self.assertEqual(tasks_at(self.tasks, Position(3, 1)), ())

    def test_marks_critical_water_and_feed_tasks(self):
        water = next(
            task
            for task in tasks_at(self.tasks, Position(0, 0))
            if task.operation == WATER
        )
        feed = next(
            task
            for task in tasks_at(self.tasks, Position(0, 1))
            if task.operation == FEED
        )

        self.assertTrue(water.critical)
        self.assertEqual(water.category, SURVIVAL)
        self.assertEqual(water.reason, "plant_will_become_weed")
        self.assertEqual(water.deadline_step, 119)

        self.assertTrue(feed.critical)
        self.assertEqual(feed.required_item, "WHEAT")
        self.assertEqual(feed.action, ["FEED"])
        self.assertEqual(feed.deadline_step, 119)
        self.assertGreater(feed.priority, water.priority)

    def test_generates_normal_daily_needs(self):
        plant_operations = {
            task.operation for task in tasks_at(self.tasks, Position(1, 0))
        }
        sheep_operations = {
            task.operation for task in tasks_at(self.tasks, Position(1, 1))
        }

        self.assertEqual(plant_operations, {WATER, HARVEST})
        self.assertEqual(sheep_operations, {FEED, CARE})

        normal_feed = next(
            task
            for task in tasks_at(self.tasks, Position(1, 1))
            if task.operation == FEED
        )
        normal_water = next(
            task
            for task in tasks_at(self.tasks, Position(1, 0))
            if task.operation == WATER
        )
        self.assertFalse(normal_feed.critical)
        self.assertFalse(normal_water.critical)
        self.assertGreater(normal_feed.priority, normal_water.priority)

    def test_does_not_harvest_an_immature_one_time_crop(self):
        raw = observation()
        raw["farms"][0]["tiles"][0][1] = plant(
            planted_day=4,
            yield_units=1,
        )

        tasks = TaskGenerator().generate(parse_observation(raw))
        operations = {
            task.operation for task in tasks_at(tasks, Position(1, 0))
        }

        self.assertEqual(operations, {WATER})

    def test_prioritizes_a_decaying_harvest(self):
        harvest = next(
            task
            for task in tasks_at(self.tasks, Position(2, 0))
            if task.operation == HARVEST
        )

        self.assertTrue(harvest.critical)
        self.assertEqual(harvest.category, PRODUCTION)
        self.assertEqual(harvest.reason, "harvest_before_decay")
        self.assertEqual(harvest.deadline_step, 90)

    def test_generates_animal_production_tasks(self):
        cow_operations = {
            task.operation for task in tasks_at(self.tasks, Position(0, 1))
        }
        goose_operations = {
            task.operation for task in tasks_at(self.tasks, Position(2, 1))
        }

        self.assertEqual(
            cow_operations,
            {FEED, HARVEST, COLLECT_FERTILIZER, CARE},
        )
        self.assertEqual(
            goose_operations,
            {HARVEST, COLLECT_FERTILIZER, CARE},
        )

    def test_generates_low_priority_weed_task(self):
        weed_task = tasks_at(self.tasks, Position(3, 0))[0]

        self.assertEqual(weed_task.operation, DIG)
        self.assertEqual(weed_task.action, ["DIG"])
        self.assertEqual(weed_task.category, MAINTENANCE)
        self.assertFalse(weed_task.critical)
        self.assertEqual(self.tasks[-1], weed_task)

    def test_tasks_are_sorted_and_have_stable_keys(self):
        priorities = [task.priority for task in self.tasks]

        self.assertEqual(priorities, sorted(priorities, reverse=True))
        self.assertEqual(len({task.key for task in self.tasks}), len(self.tasks))

    def test_urgency_increases_near_end_of_day(self):
        late = observation()
        late["step"] = 119
        late["hour"] = 23
        late_tasks = TaskGenerator().generate(parse_observation(late))

        early_water = next(
            task
            for task in self.tasks
            if task.target == Position(0, 0) and task.operation == WATER
        )
        late_water = next(
            task
            for task in late_tasks
            if task.target == Position(0, 0) and task.operation == WATER
        )

        self.assertGreater(late_water.priority, early_water.priority)
        self.assertEqual(late_water.deadline_step, 119)


if __name__ == "__main__":
    unittest.main()
