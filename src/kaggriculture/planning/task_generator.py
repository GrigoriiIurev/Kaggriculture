"""Generate prioritized farm work from a parsed Kaggriculture state.

This module decides *what* work exists, not which worker should perform it or
how that worker should reach the target.  Those decisions belong to the future
worker planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.kaggriculture.core.game_data import CROP_SPECS
from src.kaggriculture.core.state_parser import GameState, Position, TileState


SURVIVAL = "SURVIVAL"
PRODUCTION = "PRODUCTION"
MAINTENANCE = "MAINTENANCE"

WATER = "WATER"
HARVEST = "HARVEST"
FEED = "FEED"
CARE = "CARE"
COLLECT_FERTILIZER = "COLLECT_FERTILIZER"
DIG = "DIG"

ActionArgument = str | int


@dataclass(frozen=True)
class Task:
    """One piece of work that must eventually be assigned to a worker."""

    operation: str
    target: Position
    priority: int
    category: str
    reason: str
    arguments: tuple[ActionArgument, ...] = ()
    required_item: str | None = None
    deadline_step: int | None = None
    critical: bool = False
    assigned_unit: int | None = None

    @property
    def action(self) -> list[ActionArgument]:
        """Return the Kaggriculture command to use once a worker is on target."""

        return [self.operation, *self.arguments]

    @property
    def key(self) -> tuple[str, int, int, tuple[ActionArgument, ...], int | None]:
        """Stable identity used to compare a task between consecutive turns."""

        return (
            self.operation,
            self.target.x,
            self.target.y,
            self.arguments,
            self.assigned_unit,
        )


@dataclass(frozen=True)
class TaskPriorities:
    """Centralized priority values so strategy experiments can tune them."""

    critical_feed: int = 1300
    critical_water: int = 1250
    decaying_harvest: int = 1100
    daily_feed: int = 950
    daily_water: int = 900
    animal_harvest: int = 700
    plant_harvest: int = 650
    collect_fertilizer: int = 400
    care: int = 300
    clear_weed: int = 100


@dataclass(frozen=True)
class TaskGenerator:
    """Create a deterministic priority queue of work for the current farm."""

    priorities: TaskPriorities = field(default_factory=TaskPriorities)

    def generate(self, state: GameState) -> tuple[Task, ...]:
        tasks: list[Task] = []

        for plant in state.me.plants:
            tasks.extend(self._plant_tasks(plant, state))

        for animal in state.me.animals:
            tasks.extend(self._animal_tasks(animal, state))

        for weed in state.me.weeds:
            tasks.append(
                Task(
                    operation=DIG,
                    target=weed.position,
                    priority=self.priorities.clear_weed,
                    category=MAINTENANCE,
                    reason="clear_weed",
                )
            )

        return tuple(sorted(tasks, key=_task_sort_key))

    def _plant_tasks(self, plant: TileState, state: GameState) -> list[Task]:
        tasks: list[Task] = []
        deadline = _end_of_day_step(state)
        urgency = _daily_urgency_bonus(state)

        if plant.needs_water:
            critical = plant.consecutive_unwatered >= 1
            base_priority = (
                self.priorities.critical_water
                if critical
                else self.priorities.daily_water
            )
            tasks.append(
                Task(
                    operation=WATER,
                    target=plant.position,
                    priority=base_priority + urgency,
                    category=SURVIVAL,
                    reason=(
                        "plant_will_become_weed"
                        if critical
                        else "plant_needs_daily_water"
                    ),
                    deadline_step=deadline,
                    critical=critical,
                )
            )

        if plant.yield_units > 0 and _plant_is_ready_to_harvest(plant, state.day):
            decaying = (
                plant.max_lifespan_step >= 0
                and state.step >= plant.max_lifespan_step
            )
            base_priority = (
                self.priorities.decaying_harvest
                if decaying
                else self.priorities.plant_harvest
            )
            tasks.append(
                Task(
                    operation=HARVEST,
                    target=plant.position,
                    priority=base_priority + _yield_bonus(plant),
                    category=PRODUCTION,
                    reason=(
                        "harvest_before_decay"
                        if decaying
                        else "plant_has_harvest"
                    ),
                    deadline_step=plant.max_lifespan_step if decaying else None,
                    critical=decaying,
                )
            )

        return tasks

    def _animal_tasks(self, animal: TileState, state: GameState) -> list[Task]:
        tasks: list[Task] = []
        deadline = _end_of_day_step(state)
        urgency = _daily_urgency_bonus(state)

        if animal.needs_feed:
            critical = animal.consecutive_unfed >= 1
            base_priority = (
                self.priorities.critical_feed
                if critical
                else self.priorities.daily_feed
            )
            tasks.append(
                Task(
                    operation=FEED,
                    target=animal.position,
                    priority=base_priority + urgency,
                    category=SURVIVAL,
                    reason=(
                        "animal_will_escape"
                        if critical
                        else "animal_needs_daily_feed"
                    ),
                    required_item="WHEAT",
                    deadline_step=deadline,
                    critical=critical,
                )
            )

        if animal.yield_units > 0:
            tasks.append(
                Task(
                    operation=HARVEST,
                    target=animal.position,
                    priority=self.priorities.animal_harvest + _yield_bonus(animal),
                    category=PRODUCTION,
                    reason="animal_has_harvest",
                )
            )

        if animal.fertilizer_available:
            tasks.append(
                Task(
                    operation=COLLECT_FERTILIZER,
                    target=animal.position,
                    priority=self.priorities.collect_fertilizer,
                    category=PRODUCTION,
                    reason="fertilizer_available",
                )
            )

        if not animal.cared_today:
            tasks.append(
                Task(
                    operation=CARE,
                    target=animal.position,
                    priority=self.priorities.care + urgency,
                    category=PRODUCTION,
                    reason="animal_needs_daily_care",
                    deadline_step=deadline,
                )
            )

        return tasks


def _task_sort_key(task: Task) -> tuple[int, int, int, str, tuple[ActionArgument, ...]]:
    return (
        -task.priority,
        task.target.y,
        task.target.x,
        task.operation,
        task.arguments,
    )


def _daily_urgency_bonus(state: GameState) -> int:
    last_hour = max(0, state.turns_per_day - 1)
    return min(max(0, state.hour), last_hour)


def _end_of_day_step(state: GameState) -> int:
    last_hour = max(0, state.turns_per_day - 1)
    return state.step + max(0, last_hour - state.hour)


def _yield_bonus(tile: TileState) -> int:
    return min(max(0, tile.yield_units), 20)


def _plant_is_ready_to_harvest(plant: TileState, day: int) -> bool:
    if plant.crop not in CROP_SPECS or plant.planted_day is None:
        return False
    spec = CROP_SPECS[plant.crop]
    age = day - plant.planted_day
    if age < spec.first_yield_day:
        return False
    if spec.ongoing:
        return True
    return plant.yield_units >= spec.max_yield or age >= spec.max_yield_day
