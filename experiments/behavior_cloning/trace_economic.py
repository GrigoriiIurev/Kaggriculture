"""Trace the guarded learned economic policy in a full Kaggriculture game."""

from __future__ import annotations

from collections import Counter

from kaggle_environments import make

import ml_agent


def get_money(obs):
    player = obs["player"]
    return int(obs["farms"][player].get("money", 0))


def format_inventory(items):
    if not items:
        return "-"

    parts = []

    for key in sorted(items):
        value = items[key]
        if value:
            parts.append(f"{key}={value}")

    return ", ".join(parts) if parts else "-"


def operation_argument(order):
    if not order:
        return None

    op = order[0]

    if len(order) >= 2:
        return f"{op}:{order[1]}"

    return op


def main():
    # ------------------------------------------------------------
    # Wrap the real ML agent so we can record exactly what it sends
    # to Kaggriculture AFTER economic_guard.
    # ------------------------------------------------------------

    trace = []

    def traced_ml_agent(obs):
        money_before = get_money(obs)

        private = obs.get("private", {})
        shed = dict(private.get("shed", {}))
        seeds = dict(private.get("seeds", {}))

        action = ml_agent.agent(obs)

        market = action.get("market", [])

        trace.append(
            {
                "step": int(obs.get("step", 0)),
                "day": int(obs.get("day", 0)),
                "hour": int(obs.get("hour", 0)),
                "money_before": money_before,
                "shed": shed,
                "seeds": seeds,
                "market": [list(order) for order in market],
            }
        )

        return action

    # ------------------------------------------------------------
    # Run game
    # ------------------------------------------------------------

    print("Running economic trace...")
    print("ML agent:       ml_agent.py")
    print("Opponent:       behavior_agent.py")
    print()

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 720},
        debug=True,
    )

    env.run(
        [
            traced_ml_agent,
            "behavior_agent.py",
        ]
    )

    final = env.steps[-1]

    ml_reward = final[0].reward
    opponent_reward = final[1].reward

    # ------------------------------------------------------------
    # Recover money AFTER each action from the next observation.
    # ------------------------------------------------------------

    observed_money = []

    for row in trace:
        observed_money.append(row["money_before"])

    # ------------------------------------------------------------
    # Detailed trace: only turns where market was active.
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("GUARDED ECONOMIC TRACE")
    print("=" * 72)

    for i, row in enumerate(trace):
        market = row["market"]

        if not market:
            continue

        money_before = row["money_before"]

        if i + 1 < len(trace):
            money_after = trace[i + 1]["money_before"]
        else:
            money_after = money_before

        delta = money_after - money_before

        print(
            f"step {row['step']:3d} | "
            f"day {row['day']:2d} "
            f"hour {row['hour']:2d} | "
            f"money={money_before} -> {money_after} "
            f"({delta:+d})"
        )

        print(f"         market={market}")
        print(
            "         shed="
            + format_inventory(row["shed"])
        )
        print(
            "         seeds="
            + format_inventory(row["seeds"])
        )

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    market_active_turns = 0
    total_orders = 0

    operation_counts = Counter()
    pair_counts = Counter()

    buy_orders = 0
    sell_orders = 0
    hire_orders = 0
    buy_land_orders = 0

    buy_operations = {
        "BUY_PRODUCT",
        "BUY_SEED",
        "BUY_ANIMAL",
    }

    for row in trace:
        market = row["market"]

        if market:
            market_active_turns += 1

        for order in market:
            if not order:
                continue

            total_orders += 1

            op = order[0]
            operation_counts[op] += 1

            pair = operation_argument(order)

            if pair is not None:
                pair_counts[pair] += 1

            if op in buy_operations:
                buy_orders += 1
            elif op == "SELL":
                sell_orders += 1
            elif op == "HIRE":
                hire_orders += 1
            elif op == "BUY_LAND":
                buy_land_orders += 1

    starting_money = (
        trace[0]["money_before"]
        if trace
        else 0
    )

    minimum_money = (
        min(observed_money)
        if observed_money
        else 0
    )

    final_observed_money = (
        observed_money[-1]
        if observed_money
        else 0
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("ECONOMIC BC TRACE SUMMARY")
    print("=" * 72)

    print(f"Final reward:          {ml_reward}")
    print(f"Opponent reward:       {opponent_reward}")
    print(f"Starting money:        {starting_money}")
    print(f"Minimum money:         {minimum_money}")
    print(f"Final observed money:  {final_observed_money}")

    print()

    print(f"Market-active turns:   {market_active_turns}")
    print(f"Total market orders:   {total_orders}")
    print(f"Buy orders:            {buy_orders}")
    print(f"Sell orders:           {sell_orders}")
    print(f"Hire orders:           {hire_orders}")
    print(f"Buy-land orders:       {buy_land_orders}")

    print()
    print("Operations:")

    for op, count in operation_counts.most_common():
        print(f"  {op:<15} {count}")

    print()
    print("Most common operation/argument pairs:")

    for pair, count in pair_counts.most_common(20):
        print(f"  {pair:<32} {count}")

    print()
    print(
        "Important: these are the market orders AFTER "
        "economic_guard.py filtering."
    )


if __name__ == "__main__":
    main()