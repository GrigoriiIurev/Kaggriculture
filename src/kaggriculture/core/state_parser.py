"""Convert a Kaggriculture observation into convenient typed objects.

The Kaggle environment supplies a deeply nested ``obs`` object.  The parser
keeps all access to that external format in one place, so strategy code can use
expressions such as ``state.me.plants`` and ``state.market.prices``.

Example::

    state = parse_observation(obs, configuration)
    for plant in state.me.plants:
        if not plant.watered_today:
            print("Needs water:", plant.position)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any


DEFAULT_TURNS_PER_DAY = 24
DEFAULT_EPISODE_STEPS = 720
DEFAULT_SHED_CAPACITY = 100

EMPTY = "EMPTY"
LOCKED = "LOCKED"
PLANT = "PLANT"
WEED = "WEED"
COOP = "COOP"
PASTURE = "PASTURE"

KNOWN_TILE_KINDS = {PLANT, WEED, COOP, PASTURE}


class StateParseError(ValueError):
    """Raised when an observation does not match the documented format."""


@dataclass(frozen=True)
class Position:
    x: int
    y: int

    def manhattan_distance(self, other: "Position") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


@dataclass(frozen=True)
class TileState:
    position: Position
    kind: str
    crop: str | None = None
    animal: str | None = None
    planted_day: int | None = None
    placed_day: int | None = None
    watered_today: bool = False
    consecutive_unwatered: int = 0
    yield_units: int = 0
    max_lifespan_step: int = -1
    fertilized_until_day: int = -1
    fed_today: bool = False
    consecutive_unfed: int = 0
    cared_today: bool = False
    fertilizer_available: bool = False
    pending_care_bonus: int = 0

    @property
    def is_empty(self) -> bool:
        return self.kind == EMPTY

    @property
    def is_locked(self) -> bool:
        return self.kind == LOCKED

    @property
    def is_plant(self) -> bool:
        return self.kind == PLANT

    @property
    def is_weed(self) -> bool:
        return self.kind == WEED

    @property
    def is_animal_structure(self) -> bool:
        return self.kind in {COOP, PASTURE}

    @property
    def has_animal(self) -> bool:
        return self.is_animal_structure and self.animal is not None

    @property
    def needs_water(self) -> bool:
        return self.is_plant and not self.watered_today

    @property
    def needs_feed(self) -> bool:
        return self.has_animal and not self.fed_today


@dataclass(frozen=True)
class FarmState:
    money: float
    tiles: tuple[tuple[TileState, ...], ...]
    farmer: Position
    hands: tuple[Position, ...]
    unlocked_quadrants: tuple[str, ...]
    hires_today: int

    @property
    def width(self) -> int:
        return len(self.tiles[0]) if self.tiles else 0

    @property
    def height(self) -> int:
        return len(self.tiles)

    @property
    def workers(self) -> tuple[Position, ...]:
        return (self.farmer, *self.hands)

    @property
    def all_tiles(self) -> tuple[TileState, ...]:
        return tuple(tile for row in self.tiles for tile in row)

    @property
    def plants(self) -> tuple[TileState, ...]:
        return tuple(tile for tile in self.all_tiles if tile.is_plant)

    @property
    def animals(self) -> tuple[TileState, ...]:
        return tuple(tile for tile in self.all_tiles if tile.has_animal)

    @property
    def weeds(self) -> tuple[TileState, ...]:
        return tuple(tile for tile in self.all_tiles if tile.is_weed)

    @property
    def empty_tiles(self) -> tuple[TileState, ...]:
        return tuple(tile for tile in self.all_tiles if tile.is_empty)

    def tile_at(self, position: Position) -> TileState:
        if not (0 <= position.x < self.width and 0 <= position.y < self.height):
            raise IndexError(f"Position outside farm: {position}")
        return self.tiles[position.y][position.x]

    def tiles_of_kind(self, kind: str) -> tuple[TileState, ...]:
        return tuple(tile for tile in self.all_tiles if tile.kind == kind)


@dataclass(frozen=True)
class PrivateState:
    shed: dict[str, int]
    seeds: dict[str, int]
    inventories: tuple[dict[str, int], ...]
    shed_capacity: int

    @property
    def shed_used(self) -> int:
        return sum(self.shed.values())

    @property
    def shed_free(self) -> int:
        return max(0, self.shed_capacity - self.shed_used)


@dataclass(frozen=True)
class MarketState:
    inventory: dict[str, int]
    prices: dict[str, int]

    def price(self, item: str) -> int | None:
        return self.prices.get(item)


@dataclass(frozen=True)
class TownState:
    unlocked_shops: tuple[str, ...]

    @property
    def shop_counts(self) -> dict[str, int]:
        return dict(Counter(self.unlocked_shops))

    def count(self, shop: str) -> int:
        return self.shop_counts.get(shop, 0)


@dataclass(frozen=True)
class UnitState:
    index: int
    position: Position
    inventory: dict[str, int]

    @property
    def is_farmer(self) -> bool:
        return self.index == 0


@dataclass(frozen=True)
class GameState:
    player: int
    step: int
    day: int
    hour: int
    farms: tuple[FarmState, ...]
    private: PrivateState
    market: MarketState
    town: TownState
    turns_per_day: int
    episode_steps: int

    @property
    def me(self) -> FarmState:
        return self.farms[self.player]

    @property
    def opponents(self) -> tuple[FarmState, ...]:
        return tuple(farm for index, farm in enumerate(self.farms) if index != self.player)

    @property
    def opponent(self) -> FarmState:
        if len(self.opponents) != 1:
            raise ValueError("The opponent property requires exactly two players")
        return self.opponents[0]

    @property
    def units(self) -> tuple[UnitState, ...]:
        return tuple(
            UnitState(index=index, position=position, inventory=self.private.inventories[index])
            for index, position in enumerate(self.me.workers)
        )

    @property
    def remaining_steps(self) -> int:
        """Number of playable turns left, including the current turn."""

        return max(0, self.episode_steps - self.step)

    @property
    def is_last_step(self) -> bool:
        return self.step >= self.episode_steps - 1


_MISSING = object()


def parse_observation(obs: Any, configuration: Any | None = None) -> GameState:
    """Parse one raw Kaggriculture observation.

    Both normal dictionaries and Kaggle objects exposing values as attributes
    are supported.  The input is copied, so strategy code cannot accidentally
    mutate the environment's observation.
    """

    turns_per_day = _int_field(
        configuration,
        "turnsPerDay",
        "configuration",
        default=DEFAULT_TURNS_PER_DAY,
    )
    episode_steps = _int_field(
        configuration,
        "episodeSteps",
        "configuration",
        default=DEFAULT_EPISODE_STEPS,
    )
    shed_capacity = _int_field(
        configuration,
        "shedCapacity",
        "configuration",
        default=DEFAULT_SHED_CAPACITY,
    )

    player = _int_field(obs, "player", "obs")
    day = _int_field(obs, "day", "obs")
    hour = _int_field(obs, "hour", "obs")
    step = _int_field(obs, "step", "obs", default=day * turns_per_day + hour)

    raw_farms = _sequence(_field(obs, "farms", "obs"), "obs.farms")
    farms = tuple(
        _parse_farm(raw_farm, f"obs.farms[{index}]")
        for index, raw_farm in enumerate(raw_farms)
    )
    if not farms:
        raise StateParseError("obs.farms must contain at least one farm")
    if not 0 <= player < len(farms):
        raise StateParseError(
            f"obs.player={player} is outside obs.farms (size {len(farms)})"
        )

    private = _parse_private(
        _field(obs, "private", "obs"),
        shed_capacity=shed_capacity,
    )
    expected_inventories = len(farms[player].workers)
    if len(private.inventories) != expected_inventories:
        raise StateParseError(
            "obs.private.inventories must contain one inventory for the farmer "
            f"and each hand: expected {expected_inventories}, got "
            f"{len(private.inventories)}"
        )

    return GameState(
        player=player,
        step=step,
        day=day,
        hour=hour,
        farms=farms,
        private=private,
        market=_parse_market(_field(obs, "market", "obs")),
        town=_parse_town(_field(obs, "town", "obs")),
        turns_per_day=turns_per_day,
        episode_steps=episode_steps,
    )


def _parse_farm(raw: Any, path: str) -> FarmState:
    raw_rows = _sequence(_field(raw, "tiles", path), f"{path}.tiles")
    if not raw_rows:
        raise StateParseError(f"{path}.tiles must not be empty")

    tiles: list[tuple[TileState, ...]] = []
    expected_width: int | None = None
    for y, raw_row in enumerate(raw_rows):
        row_values = _sequence(raw_row, f"{path}.tiles[{y}]")
        if expected_width is None:
            expected_width = len(row_values)
        elif len(row_values) != expected_width:
            raise StateParseError(f"{path}.tiles must be rectangular")
        tiles.append(
            tuple(
                _parse_tile(raw_tile, Position(x=x, y=y), f"{path}.tiles[{y}][{x}]")
                for x, raw_tile in enumerate(row_values)
            )
        )

    if expected_width == 0:
        raise StateParseError(f"{path}.tiles rows must not be empty")

    raw_hands = _sequence(_field(raw, "hands", path), f"{path}.hands")
    raw_quadrants = _sequence(
        _field(raw, "unlocked_quadrants", path),
        f"{path}.unlocked_quadrants",
    )

    return FarmState(
        money=_number_field(raw, "money", path),
        tiles=tuple(tiles),
        farmer=_parse_position(_field(raw, "farmer", path), f"{path}.farmer"),
        hands=tuple(
            _parse_position(value, f"{path}.hands[{index}]")
            for index, value in enumerate(raw_hands)
        ),
        unlocked_quadrants=tuple(str(value) for value in raw_quadrants),
        hires_today=_int_field(raw, "hires_today", path),
    )


def _parse_tile(raw: Any, position: Position, path: str) -> TileState:
    if raw is None:
        return TileState(position=position, kind=EMPTY)
    if raw == LOCKED:
        return TileState(position=position, kind=LOCKED)
    if isinstance(raw, str):
        raise StateParseError(f"{path} has unknown string tile value {raw!r}")

    kind = str(_field(raw, "kind", path)).upper()
    if kind not in KNOWN_TILE_KINDS:
        raise StateParseError(f"{path}.kind has unknown value {kind!r}")
    if kind == WEED:
        return TileState(position=position, kind=WEED)
    if kind == PLANT:
        return TileState(
            position=position,
            kind=kind,
            crop=str(_field(raw, "crop", path)),
            planted_day=_int_field(raw, "planted_day", path),
            watered_today=_bool_field(raw, "watered_today", path),
            consecutive_unwatered=_int_field(raw, "consecutive_unwatered", path),
            yield_units=_int_field(raw, "yield_units", path),
            max_lifespan_step=_int_field(raw, "max_lifespan_step", path, default=-1),
            fertilized_until_day=_int_field(
                raw,
                "fertilized_until_day",
                path,
                default=-1,
            ),
        )

    animal = _field(raw, "animal", path, default=None)
    return TileState(
        position=position,
        kind=kind,
        animal=None if animal is None else str(animal),
        placed_day=_int_field(raw, "placed_day", path, default=-1),
        yield_units=_int_field(raw, "yield_units", path, default=0),
        fed_today=_bool_field(raw, "fed_today", path, default=False),
        consecutive_unfed=_int_field(raw, "consecutive_unfed", path, default=0),
        cared_today=_bool_field(raw, "cared_today", path, default=False),
        fertilizer_available=_bool_field(
            raw,
            "fertilizer_available",
            path,
            default=False,
        ),
        pending_care_bonus=_int_field(raw, "pending_care_bonus", path, default=0),
    )


def _parse_private(raw: Any, shed_capacity: int) -> PrivateState:
    path = "obs.private"
    raw_inventories = _sequence(
        _field(raw, "inventories", path),
        f"{path}.inventories",
    )
    return PrivateState(
        shed=_count_mapping(_field(raw, "shed", path), f"{path}.shed"),
        seeds=_count_mapping(_field(raw, "seeds", path), f"{path}.seeds"),
        inventories=tuple(
            _count_mapping(value, f"{path}.inventories[{index}]")
            for index, value in enumerate(raw_inventories)
        ),
        shed_capacity=shed_capacity,
    )


def _parse_market(raw: Any) -> MarketState:
    path = "obs.market"
    return MarketState(
        inventory=_count_mapping(_field(raw, "inventory", path), f"{path}.inventory"),
        prices=_count_mapping(_field(raw, "prices", path), f"{path}.prices"),
    )


def _parse_town(raw: Any) -> TownState:
    path = "obs.town"
    shops = _sequence(_field(raw, "unlocked_shops", path), f"{path}.unlocked_shops")
    return TownState(unlocked_shops=tuple(str(shop) for shop in shops))


def _parse_position(raw: Any, path: str) -> Position:
    values = _sequence(raw, path)
    if len(values) != 2:
        raise StateParseError(f"{path} must contain exactly [x, y]")
    return Position(x=_integer(values[0], f"{path}[0]"), y=_integer(values[1], f"{path}[1]"))


def _count_mapping(raw: Any, path: str) -> dict[str, int]:
    items = _mapping_items(raw, path)
    return {str(key): _integer(value, f"{path}.{key}") for key, value in items}


def _field(raw: Any, key: str, path: str, default: Any = _MISSING) -> Any:
    if raw is None:
        if default is not _MISSING:
            return default
        raise StateParseError(f"{path} is missing required field {key!r}")

    if isinstance(raw, Mapping):
        if key in raw:
            return raw[key]
    else:
        getter = getattr(raw, "get", None)
        if callable(getter):
            value = getter(key, _MISSING)
            if value is not _MISSING:
                return value
        if hasattr(raw, key):
            return getattr(raw, key)

    if default is not _MISSING:
        return default
    raise StateParseError(f"{path} is missing required field {key!r}")


def _sequence(raw: Any, path: str) -> Sequence[Any]:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return raw
    raise StateParseError(f"{path} must be a list or tuple")


def _mapping_items(raw: Any, path: str) -> list[tuple[Any, Any]]:
    if isinstance(raw, Mapping):
        return list(raw.items())
    items = getattr(raw, "items", None)
    if callable(items):
        return list(items())
    raise StateParseError(f"{path} must be a mapping")


def _integer(raw: Any, path: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, Real) or int(raw) != raw:
        raise StateParseError(f"{path} must be an integer, got {raw!r}")
    return int(raw)


def _int_field(raw: Any, key: str, path: str, default: Any = _MISSING) -> int:
    value = _field(raw, key, path, default=default)
    return _integer(value, f"{path}.{key}")


def _number_field(raw: Any, key: str, path: str) -> float:
    value = _field(raw, key, path)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StateParseError(f"{path}.{key} must be a number, got {value!r}")
    return float(value)


def _bool_field(raw: Any, key: str, path: str, default: Any = _MISSING) -> bool:
    value = _field(raw, key, path, default=default)
    if not isinstance(value, bool):
        raise StateParseError(f"{path}.{key} must be bool, got {value!r}")
    return value
