"""Encode Kaggriculture actions as numeric targets and decode predictions safely."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.kaggriculture.core.game_data import ANIMAL_SPECS, CROP_SPECS, PRODUCTS


WORKER_OPERATIONS = (
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PICKUP",
    "PLACE",
    "DROP",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
    "DIG",
)
MARKET_OPERATIONS = (
    "NO_ORDER",
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
    "HIRE",
    "BUY_LAND",
)
ARGUMENTS = ("NONE", *PRODUCTS, *ANIMAL_SPECS)

WORKER_OPERATION_TO_ID = {name: index for index, name in enumerate(WORKER_OPERATIONS)}
MARKET_OPERATION_TO_ID = {name: index for index, name in enumerate(MARKET_OPERATIONS)}
ARGUMENT_TO_ID = {name: index for index, name in enumerate(ARGUMENTS)}

WORKER_ITEM_OPERATIONS = {"PICKUP", "PLACE"}
WORKER_NO_ARGUMENT_OPERATIONS = set(WORKER_OPERATIONS) - WORKER_ITEM_OPERATIONS - {"PLANT"}
MARKET_NO_ARGUMENT_OPERATIONS = {"HIRE", "BUY_LAND"}


class ActionEncodingError(ValueError):
    """Raised when a recorded action does not follow the game action schema."""


@dataclass(frozen=True)
class EncodedCommand:
    operation_id: int
    argument_id: int = 0
    quantity: int = 1

    def as_dict(self) -> dict[str, int]:
        return {
            "operation_id": self.operation_id,
            "argument_id": self.argument_id,
            "quantity": self.quantity,
        }


@dataclass(frozen=True)
class EncodedAction:
    farmer: EncodedCommand
    hands: tuple[EncodedCommand, ...]
    market: tuple[EncodedCommand, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "farmer": self.farmer.as_dict(),
            "hands": [command.as_dict() for command in self.hands],
            "market": [command.as_dict() for command in self.market],
        }


class ActionEncoder:
    """Turn valid command lists into classification/regression targets."""

    def schema(self) -> dict[str, Any]:
        return {
            "version": 1,
            "worker_operations": list(WORKER_OPERATIONS),
            "market_operations": list(MARKET_OPERATIONS),
            "arguments": list(ARGUMENTS),
            "none_argument_id": ARGUMENT_TO_ID["NONE"],
            "quantity": "positive integer; ignored by operations that do not use it",
        }

    def encode_worker(self, command: Sequence[Any]) -> EncodedCommand:
        values = _command_values(command, "worker")
        operation = values[0]
        if operation not in WORKER_OPERATION_TO_ID:
            raise ActionEncodingError(f"Unknown worker operation: {operation!r}")
        if operation in WORKER_NO_ARGUMENT_OPERATIONS:
            if len(values) != 1:
                raise ActionEncodingError(f"{operation} does not accept arguments")
            return EncodedCommand(WORKER_OPERATION_TO_ID[operation])
        if len(values) < 2:
            raise ActionEncodingError(f"{operation} requires an argument")
        argument = values[1]
        if argument not in ARGUMENT_TO_ID:
            raise ActionEncodingError(f"Unknown worker argument: {argument!r}")
        if operation == "PLANT":
            if argument not in CROP_SPECS or len(values) != 2:
                raise ActionEncodingError("PLANT requires exactly one crop argument")
            return EncodedCommand(
                WORKER_OPERATION_TO_ID[operation], ARGUMENT_TO_ID[argument]
            )
        if len(values) > 3:
            raise ActionEncodingError(f"{operation} accepts at most item and quantity")
        quantity = _positive_quantity(values[2] if len(values) == 3 else 1)
        return EncodedCommand(
            WORKER_OPERATION_TO_ID[operation], ARGUMENT_TO_ID[argument], quantity
        )

    def encode_market(self, command: Sequence[Any]) -> EncodedCommand:
        values = _command_values(command, "market")
        operation = values[0]
        if operation not in MARKET_OPERATION_TO_ID or operation == "NO_ORDER":
            raise ActionEncodingError(f"Unknown market operation: {operation!r}")
        if operation in MARKET_NO_ARGUMENT_OPERATIONS:
            if len(values) != 1:
                raise ActionEncodingError(f"{operation} does not accept arguments")
            return EncodedCommand(MARKET_OPERATION_TO_ID[operation])
        if len(values) not in {2, 3}:
            raise ActionEncodingError(f"{operation} requires item and optional quantity")
        argument = values[1]
        valid_arguments = _market_arguments(operation)
        if argument not in valid_arguments:
            raise ActionEncodingError(f"{operation} cannot use argument {argument!r}")
        quantity = _positive_quantity(values[2] if len(values) == 3 else 1)
        return EncodedCommand(
            MARKET_OPERATION_TO_ID[operation], ARGUMENT_TO_ID[argument], quantity
        )

    def encode_action(
        self, action: Mapping[str, Any], expected_hands: int | None = None
    ) -> EncodedAction:
        if not isinstance(action, Mapping):
            raise ActionEncodingError("Action must be a mapping")
        farmer = self.encode_worker(action.get("farmer", ["PASS"]))
        raw_hands = action.get("hands", [])
        raw_market = action.get("market", [])
        if not _is_sequence(raw_hands) or not _is_sequence(raw_market):
            raise ActionEncodingError("hands and market must be lists")
        if expected_hands is not None and len(raw_hands) != expected_hands:
            raise ActionEncodingError(
                f"Expected {expected_hands} hand commands, got {len(raw_hands)}"
            )
        return EncodedAction(
            farmer=farmer,
            hands=tuple(self.encode_worker(command) for command in raw_hands),
            market=tuple(self.encode_market(command) for command in raw_market),
        )


class ActionDecoder:
    """Turn model outputs into valid command shapes, falling back safely."""

    def decode_worker(
        self, operation_id: int | float, argument_id: int | float = 0, quantity: int | float = 1
    ) -> list[Any]:
        operation = _name_from_prediction(WORKER_OPERATIONS, operation_id)
        if operation is None:
            return ["PASS"]
        if operation in WORKER_NO_ARGUMENT_OPERATIONS:
            return [operation]
        argument = _name_from_prediction(ARGUMENTS, argument_id)
        if operation == "PLANT":
            return [operation, argument] if argument in CROP_SPECS else ["PASS"]
        if argument not in {*PRODUCTS, *ANIMAL_SPECS}:
            return ["PASS"]
        return [operation, argument, _decoded_quantity(quantity)]

    def decode_market(
        self, operation_id: int | float, argument_id: int | float = 0, quantity: int | float = 1
    ) -> list[Any] | None:
        operation = _name_from_prediction(MARKET_OPERATIONS, operation_id)
        if operation in {None, "NO_ORDER"}:
            return None
        if operation in MARKET_NO_ARGUMENT_OPERATIONS:
            return [operation]
        argument = _name_from_prediction(ARGUMENTS, argument_id)
        if argument not in _market_arguments(operation):
            return None
        return [operation, argument, _decoded_quantity(quantity)]

    def decode_action(
        self,
        farmer: EncodedCommand,
        hands: Sequence[EncodedCommand] = (),
        market: Sequence[EncodedCommand] = (),
    ) -> dict[str, Any]:
        market_commands = [
            self.decode_market(command.operation_id, command.argument_id, command.quantity)
            for command in market
        ]
        return {
            "farmer": self.decode_worker(
                farmer.operation_id, farmer.argument_id, farmer.quantity
            ),
            "hands": [
                self.decode_worker(command.operation_id, command.argument_id, command.quantity)
                for command in hands
            ],
            "market": [command for command in market_commands if command is not None],
        }


def _command_values(command: Sequence[Any], label: str) -> list[Any]:
    if not _is_sequence(command) or not command:
        raise ActionEncodingError(f"{label} command must be a non-empty list")
    values = list(command)
    values[0] = str(values[0]).upper()
    if len(values) > 1:
        values[1] = str(values[1]).upper()
    return values


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _positive_quantity(value: Any) -> int:
    if isinstance(value, bool):
        raise ActionEncodingError("Quantity must be an integer")
    try:
        quantity = int(value)
    except (TypeError, ValueError) as exc:
        raise ActionEncodingError("Quantity must be an integer") from exc
    if quantity <= 0 or quantity != value:
        raise ActionEncodingError("Quantity must be a positive integer")
    return quantity


def _decoded_quantity(value: int | float) -> int:
    try:
        return max(1, min(100, int(round(float(value)))))
    except (TypeError, ValueError, OverflowError):
        return 1


def _name_from_prediction(names: Sequence[str], value: int | float) -> str | None:
    try:
        index = int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return None
    return names[index] if 0 <= index < len(names) else None


def _market_arguments(operation: str) -> set[str]:
    if operation == "BUY_SEED":
        return set(CROP_SPECS)
    if operation == "BUY_PRODUCT":
        return {"WHEAT", "FERTILIZER"}
    if operation == "BUY_ANIMAL":
        return set(ANIMAL_SPECS)
    if operation == "SELL":
        return set(PRODUCTS)
    return set()
