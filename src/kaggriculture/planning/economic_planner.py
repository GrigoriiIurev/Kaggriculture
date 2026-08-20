"""Plan farm expansion and market orders from the current observation.

This is intentionally a state-based policy rather than a recorded action
route.  Every call rebuilds a small budget from the farm, inventories, market,
town demand, opponent's visible farm, and time remaining in the season.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import ceil

from src.kaggriculture.core.game_data import (
    ANIMAL_SPECS,
    BASE_PRICES,
    CROP_SPECS,
    LAND_PRICES,
    PRODUCTS,
    SHOP_PRODUCTS,
)
from src.kaggriculture.core.state_parser import EMPTY, PASTURE, GameState, Position, TileState, UnitState
from src.kaggriculture.planning.task_generator import MAINTENANCE, PRODUCTION, ActionArgument, Task


BOOTSTRAP = "BOOTSTRAP"
GROWTH = "GROWTH"
REALIZE = "REALIZE"
LIQUIDATE = "LIQUIDATE"

BUILD_PASTURE = "BUILD_PASTURE"
PLACE = "PLACE"
PLANT = "PLANT"
DROP = "DROP"

MarketOrder = tuple[ActionArgument, ...]


@dataclass(frozen=True)
class EconomicConfig:
    cash_reserve: int = 250
    feed_days: int = 3
    liquidation_steps: int = 48
    last_animal_day: int = 18
    max_market_orders: int = 10
    max_animal_buys_per_turn: int = 2
    regular_sell_slots: int = 3
    seed_batch_per_crop: int = 5
    inventory_return_threshold: int = 6
    bootstrap_cows: int = 1
    bootstrap_sheep: int = 4
    growth_cows: int = 4
    growth_sheep: int = 4
    mature_cows: int = 8
    mature_sheep: int = 4
    first_land_day: int = 5
    second_land_day: int = 9


@dataclass(frozen=True)
class EconomicPlan:
    """Economic decisions to merge with ordinary farm work."""

    phase: str
    tasks: tuple[Task, ...]
    market_orders: tuple[MarketOrder, ...]
    cash_reserve: int
    target_hands: int
    target_animals: dict[str, int]
    feed_reserve: int


@dataclass(frozen=True)
class EconomicPlanner:
    """Create development tasks and a reserve-aware market queue."""

    config: EconomicConfig = field(default_factory=EconomicConfig)

    def plan(
        self,
        state: GameState,
        farm_tasks: Sequence[Task] = (),
    ) -> EconomicPlan:
        phase = _phase(state, self.config)
        animal_targets = _animal_targets(state, self.config)
        development = self._development_tasks(state, phase, animal_targets)
        transport = self._inventory_return_tasks(state, phase)
        tasks = tuple(sorted((*transport, *development), key=_task_sort_key))
        target_hands = _target_hands(state, (*farm_tasks, *tasks), phase)
        feed_days = 6 if state.day <= 7 else self.config.feed_days
        feed_reserve = _feed_reserve(state, feed_days)
        market_orders = self._market_orders(
            state,
            phase,
            animal_targets,
            target_hands,
            feed_reserve,
        )
        return EconomicPlan(
            phase=phase,
            tasks=tasks,
            market_orders=market_orders,
            cash_reserve=self.config.cash_reserve,
            target_hands=target_hands,
            target_animals=animal_targets,
            feed_reserve=feed_reserve,
        )

    def _development_tasks(
        self,
        state: GameState,
        phase: str,
        animal_targets: dict[str, int],
    ) -> list[Task]:
        if phase == LIQUIDATE:
            return []

        tasks: list[Task] = []
        target_total = sum(animal_targets.values())
        animal_slots = _animal_slots(state)[:target_total]
        animal_stock = _stored_animals(state)

        for position in animal_slots:
            tile = state.me.tile_at(position)
            if tile.kind == EMPTY:
                tasks.append(
                    Task(
                        operation=BUILD_PASTURE,
                        target=position,
                        priority=240,
                        category=MAINTENANCE,
                        reason="prepare_livestock_slot",
                    )
                )
                continue
            if tile.kind != PASTURE or tile.has_animal:
                continue
            animal = _choose_stored_animal(animal_stock, state, animal_targets)
            if animal is None:
                continue
            animal_stock[animal] -= 1
            tasks.append(
                Task(
                    operation=PLACE,
                    arguments=(animal,),
                    target=position,
                    priority=560,
                    category=PRODUCTION,
                    reason="place_purchased_animal",
                    required_item=animal,
                )
            )

        seeds = Counter(state.private.seeds)
        for position in _crop_positions(state):
            tile = state.me.tile_at(position)
            if not tile.is_empty:
                continue
            crop = _crop_for_position(position, state)
            if crop is None or seeds[crop] <= 0:
                continue
            seeds[crop] -= 1
            tasks.append(
                Task(
                    operation=PLANT,
                    arguments=(crop,),
                    target=position,
                    priority=220,
                    category=PRODUCTION,
                    reason=f"plant_{crop.lower()}_program",
                )
            )

        return tasks

    def _inventory_return_tasks(self, state: GameState, phase: str) -> list[Task]:
        tasks: list[Task] = []
        used_access: set[Position] = set()
        animal_kinds = set(ANIMAL_SPECS)
        animals_need_feed = any(animal.needs_feed for animal in state.me.animals)

        for unit in state.units:
            carried = sum(unit.inventory.values())
            if carried <= 0 or any(unit.inventory.get(item, 0) for item in animal_kinds):
                continue
            if animals_need_feed and unit.inventory.get("WHEAT", 0) > 0:
                continue
            urgent = phase in {REALIZE, LIQUIDATE}
            if not urgent and carried < self.config.inventory_return_threshold:
                continue
            access = _nearest_unused_shed_access(unit, state, used_access)
            if access is None:
                continue
            used_access.add(access)
            tasks.append(
                Task(
                    operation=DROP,
                    target=access,
                    priority=1150 if phase == LIQUIDATE else 610,
                    category=PRODUCTION,
                    reason="return_inventory_for_sale",
                    assigned_unit=unit.index,
                )
            )
        return tasks

    def _market_orders(
        self,
        state: GameState,
        phase: str,
        animal_targets: dict[str, int],
        target_hands: int,
        feed_reserve: int,
    ) -> tuple[MarketOrder, ...]:
        orders: list[MarketOrder] = []
        money = float(state.me.money)
        shed_free = state.private.shed_free
        reserve = 0 if phase == LIQUIDATE else self.config.cash_reserve

        sell_limit = (
            self.config.max_market_orders
            if phase == LIQUIDATE
            else self.config.regular_sell_slots
        )
        for item, quantity in _sale_candidates(state, phase, feed_reserve)[:sell_limit]:
            if len(orders) >= self.config.max_market_orders:
                break
            orders.append(("SELL", item, quantity))
            shed_free += quantity
            # Current price overstates a large sale; only budget 70% of it.
            money += quantity * _price(state, item) * 0.70

        if phase == LIQUIDATE:
            return tuple(orders)

        # Daily labor is cheap early in the Fibonacci curve and enables all
        # later field commitments.  Do not hire hands that would vanish soon.
        if state.hour < max(1, state.turns_per_day - 6):
            missing_hands = max(0, target_hands - len(state.me.hands))
            hire_index = state.me.hires_today
            for _ in range(missing_hands):
                cost = _fib(hire_index)
                if len(orders) >= self.config.max_market_orders or money - cost < reserve:
                    break
                orders.append(("HIRE",))
                money -= cost
                hire_index += 1

        # Feeding existing or already-purchased animals outranks new growth.
        wheat_owned = _owned_count(state, "WHEAT")
        wheat_needed = max(0, feed_reserve - wheat_owned)
        if wheat_needed > 0 and len(orders) < self.config.max_market_orders:
            unit_price = max(1, _price(state, "WHEAT"))
            survival_stock = len(state.me.animals) * 2
            spending_floor = 0 if wheat_owned < survival_stock else reserve
            affordable = max(0, int((money - spending_floor) // unit_price))
            quantity = min(wheat_needed, affordable, shed_free)
            if quantity > 0:
                orders.append(("BUY_PRODUCT", "WHEAT", quantity))
                money -= quantity * unit_price
                shed_free -= quantity

        if state.day <= self.config.last_animal_day and _animal_investment_can_mature(state):
            owned_animals = _all_animal_counts(state)
            purchases = _animal_purchases(
                state,
                animal_targets,
                owned_animals,
                self.config.max_animal_buys_per_turn,
            )
            for animal, quantity in purchases:
                if len(orders) >= self.config.max_market_orders or shed_free <= 0:
                    break
                unit_cost = ANIMAL_SPECS[animal].cost
                affordable = max(0, int((money - reserve) // unit_cost))
                quantity = min(quantity, affordable, shed_free)
                if quantity <= 0:
                    continue
                orders.append(("BUY_ANIMAL", animal, quantity))
                money -= quantity * unit_cost
                shed_free -= quantity

        desired_land = (
            1
            + int(state.day >= self.config.first_land_day)
            + int(state.day >= self.config.second_land_day)
        )
        extra_owned = max(0, len(state.me.unlocked_quadrants) - 1)
        if (
            len(state.me.unlocked_quadrants) < desired_land
            and extra_owned < len(LAND_PRICES)
            and len(orders) < self.config.max_market_orders
        ):
            land_cost = LAND_PRICES[extra_owned]
            if money - land_cost >= reserve:
                orders.append(("BUY_LAND",))
                money -= land_cost

        seed_needs = _seed_needs(state)
        for crop in _seed_purchase_order(state, seed_needs):
            if len(orders) >= self.config.max_market_orders:
                break
            missing = seed_needs[crop]
            if missing <= 0:
                continue
            unit_cost = CROP_SPECS[crop].seed_cost
            affordable = max(0, int((money - reserve) // unit_cost))
            quantity = min(missing, affordable, self.config.seed_batch_per_crop)
            if quantity <= 0:
                continue
            orders.append(("BUY_SEED", crop, quantity))
            money -= quantity * unit_cost

        return tuple(orders[: self.config.max_market_orders])


def _phase(state: GameState, config: EconomicConfig) -> str:
    if state.remaining_steps <= config.liquidation_steps:
        return LIQUIDATE
    if state.day <= 4:
        return BOOTSTRAP
    if state.day <= 21:
        return GROWTH
    return REALIZE


def _animal_targets(state: GameState, config: EconomicConfig) -> dict[str, int]:
    if state.day <= 4:
        return {"COW": config.bootstrap_cows, "SHEEP": config.bootstrap_sheep}
    if state.day <= 7:
        return {"COW": config.growth_cows, "SHEEP": config.growth_sheep}
    return {"COW": config.mature_cows, "SHEEP": config.mature_sheep}


def _animal_slots(state: GameState) -> tuple[Position, ...]:
    half_x = state.me.width // 2
    half_y = state.me.height // 2
    shed_corner = Position(max(0, half_x - 1), max(0, half_y - 1))
    nw = [
        tile.position
        for tile in state.me.all_tiles
        if tile.position.x < half_x and tile.position.y < half_y and not tile.is_locked
    ]
    return tuple(
        sorted(
            nw,
            key=lambda position: (
                position.manhattan_distance(shed_corner),
                -position.y,
                -position.x,
            ),
        )[:12]
    )


def _crop_positions(state: GameState) -> tuple[Position, ...]:
    animal_slots = set(_animal_slots(state))
    candidates = [
        tile.position
        for tile in state.me.all_tiles
        if not tile.is_locked and tile.position not in animal_slots
    ]
    access = _shed_access_positions(state)
    limit = 13 + 12 * max(0, len(state.me.unlocked_quadrants) - 1)
    return tuple(
        sorted(
            candidates,
            key=lambda position: (
                min(position.manhattan_distance(point) for point in access),
                position.y,
                position.x,
            ),
        )[:limit]
    )


def _crop_for_position(position: Position, state: GameState) -> str | None:
    feasible = [
        crop
        for crop, spec in CROP_SPECS.items()
        if _remaining_days(state) >= spec.first_yield_day + 1
    ]
    if not feasible:
        return None

    role = (position.x * 3 + position.y * 5) % 4
    preferred = "WHEAT" if role == 0 else ("STRAWBERRY" if role == 1 else "MELON")
    if preferred in feasible:
        if preferred in {"STRAWBERRY", "MELON"}:
            alternative = "MELON" if preferred == "STRAWBERRY" else "STRAWBERRY"
            if alternative in feasible:
                preferred_score = _crop_score(preferred, state)
                alternative_score = _crop_score(alternative, state)
                if alternative_score > preferred_score * 1.35:
                    return alternative
        return preferred
    return max(feasible, key=lambda crop: (_crop_score(crop, state), crop))


def _crop_score(crop: str, state: GameState) -> float:
    spec = CROP_SPECS[crop]
    expected_yield = 4 if spec.ongoing else min(spec.max_yield, 1 + spec.max_yield_day // 2 + 1)
    gross = expected_yield * _price(state, crop) - spec.seed_cost
    demand = _town_demand_weight(state, crop)
    own = sum(plant.crop == crop for plant in state.me.plants)
    opponent = sum(plant.crop == crop for plant in state.opponent.plants)
    crowding = 1.0 + 0.025 * own + 0.04 * opponent
    return max(0.0, gross) * (1.0 + 0.08 * demand) / crowding


def _seed_needs(state: GameState) -> Counter[str]:
    desired: Counter[str] = Counter()
    for position in _crop_positions(state):
        if state.me.tile_at(position).is_empty:
            crop = _crop_for_position(position, state)
            if crop is not None:
                desired[crop] += 1
    for crop, count in state.private.seeds.items():
        desired[crop] = max(0, desired[crop] - count)
    return desired


def _seed_purchase_order(state: GameState, needs: Counter[str]) -> list[str]:
    return sorted(
        (crop for crop, need in needs.items() if need > 0),
        key=lambda crop: (-_crop_score(crop, state), crop),
    )


def _stored_animals(state: GameState) -> Counter[str]:
    counts = Counter({animal: state.private.shed.get(animal, 0) for animal in ANIMAL_SPECS})
    for inventory in state.private.inventories:
        for animal in ANIMAL_SPECS:
            counts[animal] += inventory.get(animal, 0)
    return counts


def _all_animal_counts(state: GameState) -> Counter[str]:
    counts = _stored_animals(state)
    for tile in state.me.animals:
        if tile.animal is not None:
            counts[tile.animal] += 1
    return counts


def _choose_stored_animal(
    stock: Counter[str],
    state: GameState,
    targets: dict[str, int],
) -> str | None:
    placed = Counter(tile.animal for tile in state.me.animals if tile.animal)
    options = [animal for animal, count in stock.items() if count > 0]
    if not options:
        return None
    return max(
        options,
        key=lambda animal: (
            targets.get(animal, 0) - placed[animal],
            _animal_score(animal, state),
            animal,
        ),
    )


def _animal_purchases(
    state: GameState,
    targets: dict[str, int],
    owned: Counter[str],
    limit: int,
) -> list[tuple[str, int]]:
    selected: Counter[str] = Counter()
    for _ in range(limit):
        options = [
            animal
            for animal, target in targets.items()
            if owned[animal] + selected[animal] < target
        ]
        if not options:
            break
        animal = max(
            options,
            key=lambda item: (
                _animal_score(item, state),
                targets[item] - owned[item] - selected[item],
                item,
            ),
        )
        selected[animal] += 1
    return sorted(selected.items(), key=lambda pair: -_animal_score(pair[0], state))


def _animal_score(animal: str, state: GameState) -> float:
    spec = ANIMAL_SPECS[animal]
    productive_days = max(0, _remaining_days(state) - spec.first_yield_day)
    cycles = productive_days // spec.interval + int(productive_days > 0)
    gross = cycles * 2 * _price(state, spec.product)
    demand = _town_demand_weight(state, spec.product)
    opponent_count = sum(tile.animal == animal for tile in state.opponent.animals)
    return gross * (1.0 + 0.08 * demand) / (spec.cost * (1.0 + 0.05 * opponent_count))


def _sale_candidates(
    state: GameState,
    phase: str,
    feed_reserve: int,
) -> list[tuple[str, int]]:
    pressure = state.private.shed_used >= int(state.private.shed_capacity * 0.75)
    candidates: list[tuple[str, int, float]] = []
    fragility = {
        "MELON": 5,
        "WOOL": 5,
        "MILK": 4,
        "STRAWBERRY": 4,
        "TOMATO": 2,
        "EGG": 2,
        "CARROT": 1,
        "WHEAT": 1,
        "FERTILIZER": 1,
    }
    batch = {
        "MELON": 6,
        "WOOL": 6,
        "MILK": 8,
        "STRAWBERRY": 8,
        "TOMATO": 12,
        "EGG": 12,
        "CARROT": 16,
        "WHEAT": 20,
        "FERTILIZER": 20,
    }

    for item in PRODUCTS:
        available = state.private.shed.get(item, 0)
        if item == "WHEAT" and phase != LIQUIDATE:
            available = max(0, available - feed_reserve)
        if available <= 0:
            continue
        price = _price(state, item)
        ratio = price / BASE_PRICES[item]
        if phase != LIQUIDATE and ratio < 0.30 and not pressure:
            continue
        quantity = available if phase == LIQUIDATE or pressure else min(available, batch[item])
        priority = fragility[item] * 10 + ratio + 0.25 * _town_demand_weight(state, item)
        candidates.append((item, quantity, priority))

    candidates.sort(key=lambda value: (-value[2], value[0]))
    return [(item, quantity) for item, quantity, _ in candidates]


def _feed_reserve(state: GameState, days: int) -> int:
    placed = len(state.me.animals)
    pipeline = sum(_stored_animals(state).values())
    return (placed + pipeline) * days


def _owned_count(state: GameState, item: str) -> int:
    return state.private.shed.get(item, 0) + sum(
        inventory.get(item, 0) for inventory in state.private.inventories
    )


def _town_demand_weight(state: GameState, item: str) -> int:
    weight = 1 if item != "FERTILIZER" else 0
    for shop in state.town.unlocked_shops:
        products = SHOP_PRODUCTS.get(shop, ())
        if item in products:
            weight += 2 if len(products) == 1 else 1
    return weight


def _target_hands(
    state: GameState,
    tasks: Sequence[Task],
    phase: str,
) -> int:
    if phase == LIQUIDATE:
        floor = 2
    else:
        floor = 4
    critical = sum(task.critical for task in tasks)
    workload_target = ceil((len(tasks) + critical * 2) / 7)
    return min(7, max(floor, workload_target))


def _remaining_days(state: GameState) -> int:
    return ceil(state.remaining_steps / max(1, state.turns_per_day))


def _animal_investment_can_mature(state: GameState) -> bool:
    return _remaining_days(state) >= max(
        ANIMAL_SPECS["COW"].first_yield_day,
        ANIMAL_SPECS["SHEEP"].first_yield_day,
    ) + 2


def _price(state: GameState, item: str) -> int:
    return max(1, state.market.prices.get(item, BASE_PRICES[item]))


def _fib(index: int) -> int:
    a, b = 1, 1
    for _ in range(max(0, index)):
        a, b = b, a + b
    return a


def _shed_access_positions(state: GameState) -> tuple[Position, ...]:
    center_x = state.me.width // 2
    center_y = state.me.height // 2
    return (
        Position(center_x - 1, center_y - 1),
        Position(center_x, center_y - 1),
        Position(center_x - 1, center_y),
        Position(center_x, center_y),
    )


def _nearest_unused_shed_access(
    unit: UnitState,
    state: GameState,
    used: set[Position],
) -> Position | None:
    candidates = [position for position in _shed_access_positions(state) if position not in used]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda position: (
            unit.position.manhattan_distance(position),
            position.y,
            position.x,
        ),
    )


def _task_sort_key(task: Task) -> tuple[int, int, int, str]:
    return (-task.priority, task.target.y, task.target.x, task.operation)
