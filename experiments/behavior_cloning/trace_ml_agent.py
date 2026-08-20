"""Trace Economic BC decisions during one Kaggriculture game."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle_environments import make


DEFAULT_ML_AGENT = "ml_agent.py"
DEFAULT_OPPONENT = "behavior_agent.py"
DEFAULT_OUTPUT = "ml_economic_trace.json"


def safe_number(value, default=0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_player_observation(step_state, player: int):
    obs = step_state[player].observation

    # Kaggle observation objects behave like dicts,
    # but converting makes later processing simpler.
    return dict(obs)


def extract_snapshot(
    step_index: int,
    step_state,
    player: int,
) -> dict:
    obs = get_player_observation(
        step_state,
        player,
    )

    farms = obs.get("farms", [])
    private = obs.get("private", {})
    market = obs.get("market", {})

    if player < len(farms):
        farm = farms[player]
    else:
        farm = {}

    money = safe_number(
        farm.get("money", 0)
    )

    shed = dict(
        private.get("shed", {})
    )

    seeds = dict(
        private.get("seeds", {})
    )

    prices = dict(
        market.get("prices", {})
    )

    inventories = private.get(
        "inventories",
        [],
    )

    carried = {}

    for inventory in inventories:
        if not isinstance(inventory, dict):
            continue

        for item, quantity in inventory.items():
            carried[item] = (
                carried.get(item, 0)
                + int(quantity)
            )

    return {
        "step": step_index,
        "day": obs.get("day"),
        "hour": obs.get("hour"),
        "money": money,
        "shed": shed,
        "seeds": seeds,
        "carried": carried,
        "prices": prices,
    }


def extract_action(
    steps,
    step_index: int,
    player: int,
):
    """
    Kaggle stores the action chosen from state t
    in the following frame t + 1.
    """

    next_index = step_index + 1

    if next_index >= len(steps):
        return None

    action = steps[next_index][player].action

    if action is None:
        return None

    return action


def money_change(
    snapshots: list[dict],
    index: int,
) -> float | None:
    if index + 1 >= len(snapshots):
        return None

    return (
        snapshots[index + 1]["money"]
        - snapshots[index]["money"]
    )


def important_inventory(
    inventory: dict,
) -> str:
    nonzero = {
        key: value
        for key, value in inventory.items()
        if value
    }

    if not nonzero:
        return "-"

    return ", ".join(
        f"{key}={value}"
        for key, value in sorted(
            nonzero.items()
        )
    )


def print_market_action(
    snapshot: dict,
    action,
    delta_money,
) -> None:
    if not isinstance(action, dict):
        market = []
    else:
        market = action.get(
            "market",
            [],
        ) or []

    # Only print turns where something interesting
    # happened economically.
    if (
        not market
        and not delta_money
    ):
        return

    step = snapshot["step"]
    day = snapshot["day"]
    hour = snapshot["hour"]
    money = snapshot["money"]

    if delta_money is None:
        delta_text = ""
    else:
        delta_text = (
            f" -> {money + delta_money:.0f} "
            f"({delta_money:+.0f})"
        )

    print(
        f"step {step:>3} | "
        f"day {str(day):>2} "
        f"hour {str(hour):>2} | "
        f"money={money:.0f}{delta_text}"
    )

    print(
        f"         market={market}"
    )

    if market:
        print(
            "         shed="
            + important_inventory(
                snapshot["shed"]
            )
        )

        print(
            "         seeds="
            + important_inventory(
                snapshot["seeds"]
            )
        )


def summarize(
    trace: list[dict],
    final_reward: float,
    opponent_reward: float,
) -> None:
    market_turns = 0
    total_orders = 0

    operation_counts = {}
    argument_counts = {}

    buys = 0
    sells = 0
    hires = 0
    land = 0

    for row in trace:
        action = row.get("action")

        if not isinstance(action, dict):
            continue

        market = action.get(
            "market",
            [],
        ) or []

        if market:
            market_turns += 1

        total_orders += len(market)

        for order in market:
            if not order:
                continue

            operation = str(order[0])

            operation_counts[operation] = (
                operation_counts.get(
                    operation,
                    0,
                )
                + 1
            )

            if len(order) >= 2:
                argument = str(order[1])

                key = (
                    f"{operation}:{argument}"
                )

                argument_counts[key] = (
                    argument_counts.get(
                        key,
                        0,
                    )
                    + 1
                )

            if operation.startswith("BUY_"):
                buys += 1

            elif operation == "SELL":
                sells += 1

            elif operation == "HIRE":
                hires += 1

            elif operation == "BUY_LAND":
                land += 1

    money_values = [
        row["state"]["money"]
        for row in trace
    ]

    print()
    print("=" * 72)
    print("ECONOMIC BC TRACE SUMMARY")
    print("=" * 72)

    print(
        f"Final reward:       {final_reward:.0f}"
    )

    print(
        f"Opponent reward:    {opponent_reward:.0f}"
    )

    if money_values:
        print(
            f"Starting money:     "
            f"{money_values[0]:.0f}"
        )

        print(
            f"Minimum money:      "
            f"{min(money_values):.0f}"
        )

        print(
            f"Final observed money: "
            f"{money_values[-1]:.0f}"
        )

    print()
    print(
        f"Market-active turns: {market_turns}"
    )

    print(
        f"Total market orders: {total_orders}"
    )

    print(
        f"Buy orders:          {buys}"
    )

    print(
        f"Sell orders:         {sells}"
    )

    print(
        f"Hire orders:         {hires}"
    )

    print(
        f"Buy-land orders:     {land}"
    )

    print()
    print("Operations:")

    for operation, count in sorted(
        operation_counts.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    ):
        print(
            f"  {operation:<15} {count}"
        )

    print()
    print("Most common operation/argument pairs:")

    for key, count in sorted(
        argument_counts.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )[:20]:
        print(
            f"  {key:<30} {count}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ml-agent",
        default=DEFAULT_ML_AGENT,
    )

    parser.add_argument(
        "--opponent",
        default=DEFAULT_OPPONENT,
    )

    parser.add_argument(
        "--player",
        type=int,
        choices=[0, 1],
        default=0,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--all-turns",
        action="store_true",
        help=(
            "Print every turn instead of only "
            "economically active turns."
        ),
    )

    args = parser.parse_args()

    if args.player == 0:
        agents = [
            args.ml_agent,
            args.opponent,
        ]
    else:
        agents = [
            args.opponent,
            args.ml_agent,
        ]

    print("Running traced game...")
    print(
        f"ML agent: {args.ml_agent} "
        f"(player {args.player})"
    )
    print(
        f"Opponent: {args.opponent}"
    )
    print()

    env = make(
        "kaggriculture",
        configuration={
            "episodeSteps": 720,
        },
        debug=False,
    )

    env.run(agents)

    steps = env.steps

    snapshots = [
        extract_snapshot(
            step_index,
            step_state,
            args.player,
        )
        for step_index, step_state
        in enumerate(steps)
    ]

    trace = []

    for step_index, snapshot in enumerate(
        snapshots
    ):
        action = extract_action(
            steps,
            step_index,
            args.player,
        )

        delta = money_change(
            snapshots,
            step_index,
        )

        row = {
            "state": snapshot,
            "action": action,
            "money_change": delta,
        }

        trace.append(row)

        if args.all_turns:
            print_market_action(
                snapshot,
                action,
                delta,
            )
        else:
            market = []

            if isinstance(action, dict):
                market = (
                    action.get(
                        "market",
                        [],
                    )
                    or []
                )

            if market:
                print_market_action(
                    snapshot,
                    action,
                    delta,
                )

    final = steps[-1]

    ml_reward = safe_number(
        final[args.player].reward
    )

    opponent_player = (
        1 - args.player
    )

    opponent_reward = safe_number(
        final[opponent_player].reward
    )

    output = {
        "ml_agent": args.ml_agent,
        "opponent": args.opponent,
        "player": args.player,
        "reward": ml_reward,
        "opponent_reward": opponent_reward,
        "trace": trace,
    }

    output_path = Path(
        args.output
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.write("\n")

    summarize(
        trace,
        ml_reward,
        opponent_reward,
    )

    print()
    print(
        f"Full trace saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()