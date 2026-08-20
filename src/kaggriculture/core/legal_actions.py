"""State-aware legal choices used to mask worker model predictions."""

from __future__ import annotations

from src.kaggriculture.core.game_data import ANIMAL_SPECS, CROP_SPECS
from src.kaggriculture.core.state_parser import COOP, EMPTY, LOCKED, PASTURE, PLANT, WEED, GameState


ARGUMENT_OPERATIONS = {"PICKUP", "PLACE", "PLANT"}


def legal_worker_operations(state: GameState, worker_index: int) -> set[str]:
    worker = state.units[worker_index]
    tile = state.me.tile_at(worker.position)
    operations = {"PASS"}
    if worker.position.y > 0:
        operations.add("NORTH")
    if worker.position.y + 1 < state.me.height:
        operations.add("SOUTH")
    if worker.position.x > 0:
        operations.add("WEST")
    if worker.position.x + 1 < state.me.width:
        operations.add("EAST")

    shed_adjacent = _is_shed_adjacent(
        worker.position.x, worker.position.y, state.me.width, state.me.height
    )
    if shed_adjacent and any(state.private.shed.values()):
        operations.add("PICKUP")
    if shed_adjacent and any(worker.inventory.values()):
        operations.update({"DROP", "PLACE"})

    if tile.kind == EMPTY:
        if tile.kind != LOCKED:
            if any(state.private.seeds.get(crop, 0) > 0 for crop in CROP_SPECS):
                operations.add("PLANT")
            operations.update({"BUILD_COOP", "BUILD_PASTURE"})
    elif tile.kind == PLANT:
        if not tile.watered_today:
            operations.add("WATER")
        if tile.yield_units > 0:
            operations.add("HARVEST")
        if worker.inventory.get("FERTILIZER", 0) > 0:
            operations.add("FERTILIZE")
        operations.add("DIG")
    elif tile.kind == WEED:
        operations.add("DIG")
    elif tile.kind in {COOP, PASTURE}:
        if tile.has_animal:
            if tile.yield_units > 0:
                operations.add("HARVEST")
            if not tile.fed_today and worker.inventory.get("WHEAT", 0) > 0:
                operations.add("FEED")
            if tile.fertilizer_available:
                operations.add("COLLECT_FERTILIZER")
            if not tile.cared_today:
                operations.add("CARE")
        else:
            operations.add("DIG")
            if _matching_carried_animals(state, worker_index, tile.kind):
                operations.add("PLACE")
    return operations


def legal_worker_arguments(
    state: GameState, worker_index: int, operation: str
) -> set[str]:
    worker = state.units[worker_index]
    if operation == "PICKUP":
        return {item for item, count in state.private.shed.items() if count > 0}
    if operation == "PLANT":
        return {
            crop for crop in CROP_SPECS if state.private.seeds.get(crop, 0) > 0
        }
    if operation == "PLACE":
        tile = state.me.tile_at(worker.position)
        if tile.kind in {COOP, PASTURE}:
            return _matching_carried_animals(state, worker_index, tile.kind)
        if _is_shed_adjacent(
            worker.position.x,
            worker.position.y,
            state.me.width,
            state.me.height,
        ):
            return {item for item, count in worker.inventory.items() if count > 0}
    return set()


def _matching_carried_animals(
    state: GameState, worker_index: int, structure: str
) -> set[str]:
    inventory = state.units[worker_index].inventory
    return {
        animal
        for animal, spec in ANIMAL_SPECS.items()
        if spec.structure == structure and inventory.get(animal, 0) > 0
    }


def _is_shed_adjacent(x: int, y: int, width: int, height: int) -> bool:
    center_x = {width // 2 - 1, width // 2}
    center_y = {height // 2 - 1, height // 2}
    return x in center_x and y in center_y
