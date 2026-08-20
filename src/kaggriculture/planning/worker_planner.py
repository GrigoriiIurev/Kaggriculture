"""Assign farm tasks to workers and produce one command per worker."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from src.kaggriculture.core.state_parser import FarmState, GameState, Position, UnitState
from src.kaggriculture.planning.task_generator import ActionArgument, Task


IDLE = "IDLE"
MOVE_TO_SHED = "MOVE_TO_SHED"
PICKUP_ITEM = "PICKUP_ITEM"
MOVE_TO_TASK = "MOVE_TO_TASK"
EXECUTE_TASK = "EXECUTE_TASK"

PASS = "PASS"
NORTH = "NORTH"
SOUTH = "SOUTH"
EAST = "EAST"
WEST = "WEST"
PICKUP = "PICKUP"

Command = tuple[ActionArgument, ...]
TaskKey = tuple[str, int, int, tuple[ActionArgument, ...], int | None]


@dataclass(frozen=True)
class UnitDecision:
    """The command and current route phase selected for one worker."""

    unit_index: int
    command: Command
    phase: str
    task: Task | None = None
    waypoint: Position | None = None
    estimated_steps: int = 0


@dataclass(frozen=True)
class WorkerPlan:
    """A complete set of worker commands for one Kaggriculture turn."""

    decisions: tuple[UnitDecision, ...]

    @property
    def assigned_tasks(self) -> tuple[Task, ...]:
        return tuple(
            decision.task
            for decision in self.decisions
            if decision.task is not None
        )

    @property
    def action(self) -> dict[str, list]:
        if not self.decisions:
            return {"farmer": [PASS], "hands": [], "market": []}

        ordered = tuple(sorted(self.decisions, key=lambda decision: decision.unit_index))
        return {
            "farmer": list(ordered[0].command),
            "hands": [list(decision.command) for decision in ordered[1:]],
            "market": [],
        }


@dataclass(frozen=True)
class _Route:
    estimated_steps: int
    pickup_at: Position | None = None


@dataclass
class _PlannerMemory:
    last_step: int = -1
    last_day: int = -1
    assignments: dict[int, TaskKey] = field(default_factory=dict)


@dataclass
class WorkerPlanner:
    """Greedily assign high-priority tasks while keeping routes stable."""

    continuity_bonus: int = 3
    _memories: dict[int, _PlannerMemory] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def plan(self, state: GameState, tasks: Sequence[Task]) -> WorkerPlan:
        memory = self._prepare_memory(state)
        assignments = self._assign_tasks(state, tasks, memory)

        decisions = tuple(
            self._make_decision(unit, assignments.get(unit.index), state)
            for unit in state.units
        )
        memory.assignments = {
            decision.unit_index: decision.task.key
            for decision in decisions
            if decision.task is not None
        }
        return WorkerPlan(decisions=decisions)

    def reset(self) -> None:
        """Forget all route continuity information."""

        self._memories.clear()

    def _prepare_memory(self, state: GameState) -> _PlannerMemory:
        memory = self._memories.setdefault(state.player, _PlannerMemory())
        new_episode = state.step == 0 or state.step < memory.last_step
        new_day = memory.last_day >= 0 and state.day != memory.last_day
        if new_episode or new_day:
            memory.assignments.clear()
        memory.last_step = state.step
        memory.last_day = state.day
        return memory

    def _assign_tasks(
        self,
        state: GameState,
        tasks: Sequence[Task],
        memory: _PlannerMemory,
    ) -> dict[int, tuple[Task, _Route]]:
        available_units = {unit.index: unit for unit in state.units}
        shed_remaining = dict(state.private.shed)
        occupied_targets: set[Position] = set()
        assignments: dict[int, tuple[Task, _Route]] = {}

        for task in sorted(tasks, key=_assignment_task_key):
            if task.target in occupied_targets or not _is_on_farm(task.target, state.me):
                continue

            candidates: list[tuple[int, int, int, UnitState, _Route]] = []
            for unit in available_units.values():
                if task.assigned_unit is not None and task.assigned_unit != unit.index:
                    continue
                route = _route_for(unit, task, state, shed_remaining)
                if route is None:
                    continue
                continuity = (
                    self.continuity_bonus
                    if memory.assignments.get(unit.index) == task.key
                    else 0
                )
                candidates.append(
                    (
                        route.estimated_steps - continuity,
                        route.estimated_steps,
                        unit.index,
                        unit,
                        route,
                    )
                )

            if not candidates:
                continue

            _, _, unit_index, unit, route = min(candidates, key=lambda value: value[:3])
            assignments[unit_index] = (task, route)
            available_units.pop(unit_index)
            occupied_targets.add(task.target)

            if route.pickup_at is not None and task.required_item is not None:
                shed_remaining[task.required_item] = max(
                    0,
                    shed_remaining.get(task.required_item, 0) - 1,
                )

            if not available_units:
                break

        return assignments

    def _make_decision(
        self,
        unit: UnitState,
        assignment: tuple[Task, _Route] | None,
        state: GameState,
    ) -> UnitDecision:
        if assignment is None:
            return UnitDecision(
                unit_index=unit.index,
                command=(PASS,),
                phase=IDLE,
            )

        task, route = assignment
        has_required_item = (
            task.required_item is None
            or unit.inventory.get(task.required_item, 0) > 0
        )

        if not has_required_item:
            if route.pickup_at is None:
                return UnitDecision(
                    unit_index=unit.index,
                    command=(PASS,),
                    phase=IDLE,
                )
            if unit.position == route.pickup_at:
                return UnitDecision(
                    unit_index=unit.index,
                    command=(PICKUP, task.required_item, 1),
                    phase=PICKUP_ITEM,
                    task=task,
                    waypoint=route.pickup_at,
                    estimated_steps=route.estimated_steps,
                )
            return UnitDecision(
                unit_index=unit.index,
                command=_move_toward(unit.position, route.pickup_at),
                phase=MOVE_TO_SHED,
                task=task,
                waypoint=route.pickup_at,
                estimated_steps=route.estimated_steps,
            )

        if unit.position == task.target:
            return UnitDecision(
                unit_index=unit.index,
                command=tuple(task.action),
                phase=EXECUTE_TASK,
                task=task,
                waypoint=task.target,
                estimated_steps=route.estimated_steps,
            )

        return UnitDecision(
            unit_index=unit.index,
            command=_move_toward(unit.position, task.target),
            phase=MOVE_TO_TASK,
            task=task,
            waypoint=task.target,
            estimated_steps=route.estimated_steps,
        )


def _assignment_task_key(task: Task) -> tuple[int, int, int, int, str]:
    deadline = task.deadline_step if task.deadline_step is not None else 10**9
    return (
        -task.priority,
        deadline,
        task.target.y,
        task.target.x,
        task.operation,
    )


def _route_for(
    unit: UnitState,
    task: Task,
    state: GameState,
    shed_remaining: dict[str, int],
) -> _Route | None:
    if task.required_item is None or unit.inventory.get(task.required_item, 0) > 0:
        return _Route(
            estimated_steps=unit.position.manhattan_distance(task.target) + 1,
        )

    if shed_remaining.get(task.required_item, 0) <= 0:
        return None

    pickup_at = min(
        _shed_access_positions(state.me),
        key=lambda position: (
            unit.position.manhattan_distance(position)
            + position.manhattan_distance(task.target),
            unit.position.manhattan_distance(position),
            position.y,
            position.x,
        ),
    )
    estimated_steps = (
        unit.position.manhattan_distance(pickup_at)
        + 1
        + pickup_at.manhattan_distance(task.target)
        + 1
    )
    return _Route(estimated_steps=estimated_steps, pickup_at=pickup_at)


def _shed_access_positions(farm: FarmState) -> tuple[Position, ...]:
    center_x = farm.width // 2
    center_y = farm.height // 2
    candidates = (
        Position(center_x - 1, center_y - 1),
        Position(center_x, center_y - 1),
        Position(center_x - 1, center_y),
        Position(center_x, center_y),
    )
    return tuple(position for position in candidates if _is_on_farm(position, farm))


def _is_on_farm(position: Position, farm: FarmState) -> bool:
    return 0 <= position.x < farm.width and 0 <= position.y < farm.height


def _move_toward(current: Position, target: Position) -> Command:
    if current.x < target.x:
        return (EAST,)
    if current.x > target.x:
        return (WEST,)
    if current.y < target.y:
        return (SOUTH,)
    if current.y > target.y:
        return (NORTH,)
    return (PASS,)
