"""Safety/execution guard for the learned economic policy."""

from __future__ import annotations

from src.kaggriculture.core.game_data import ANIMAL_SPECS, CROP_SPECS


# Conservative limits.
MAX_MARKET_ORDERS = 10
MAX_HIRES_PER_TURN = 2

# Do not let the learned policy burn the whole bank on ordinary purchases.
CASH_RESERVE = 100

# Avoid repeated BUY_PRODUCT WHEAT predictions when the farm already
# has a reasonable stock.
WHEAT_TARGET = 20


def guard_market_orders(obs, orders):
    """
    Filter Economic BC predictions before sending them to Kaggriculture.

    The guard does not choose an economic strategy. It only removes
    obviously impossible, duplicated, or excessively risky orders.
    """

    player = obs["player"]
    farm = obs["farms"][player]
    private = obs["private"]

    money = int(farm.get("money", 0))
    shed = private.get("shed", {})
    seeds = private.get("seeds", {})

    unlocked = set(farm.get("unlocked_quadrants", []))
    hires_today = int(farm.get("hires_today", 0))

    accepted = []

    hires_added = 0
    land_added = 0

    # Approximate remaining budget while processing the predicted
    # sequence from left to right.
    budget = money

    for order in orders:
        if len(accepted) >= MAX_MARKET_ORDERS:
            break

        if not order:
            continue

        op = order[0]

        # ------------------------------------------------------------
        # HIRE
        # ------------------------------------------------------------

        if op == "HIRE":
            if hires_added >= MAX_HIRES_PER_TURN:
                continue

            n = hires_today + hires_added

            # fib(0), fib(1), fib(2), ...
            cost = _fib(n)

            if cost > budget:
                continue

            accepted.append(["HIRE"])
            budget -= cost
            hires_added += 1
            continue

        # ------------------------------------------------------------
        # BUY_LAND
        # ------------------------------------------------------------

        if op == "BUY_LAND":
            # At most one land purchase per turn.
            if land_added:
                continue

            locked_count = 4 - len(unlocked)

            if locked_count <= 0:
                continue

            land_cost = _next_land_cost(unlocked)

            if land_cost is None:
                continue

            if budget - land_cost < CASH_RESERVE:
                continue

            accepted.append(["BUY_LAND"])
            budget -= land_cost
            land_added += 1
            continue

        # ------------------------------------------------------------
        # BUY_PRODUCT
        # ------------------------------------------------------------

        if op == "BUY_PRODUCT":
            if len(order) < 3:
                continue

            item = order[1]
            quantity = _positive_int(order[2])

            if quantity <= 0:
                continue

            # Kaggriculture only allows WHEAT and FERTILIZER here.
            if item not in {"WHEAT", "FERTILIZER"}:
                continue

            if item == "WHEAT":
                current = int(shed.get("WHEAT", 0))

                # Do not repeatedly buy wheat when enough is already stored.
                missing = max(0, WHEAT_TARGET - current)

                if missing <= 0:
                    continue

                quantity = min(quantity, missing)

            price = _market_price(obs, item)

            if price is None:
                continue

            affordable = max(
                0,
                (budget - CASH_RESERVE) // price,
            )

            quantity = min(quantity, affordable)

            if quantity <= 0:
                continue

            accepted.append(
                ["BUY_PRODUCT", item, quantity]
            )

            budget -= price * quantity
            continue

        # ------------------------------------------------------------
        # BUY_SEED
        # ------------------------------------------------------------

        if op == "BUY_SEED":
            if len(order) < 3:
                continue

            crop = order[1]
            quantity = _positive_int(order[2])

            if quantity <= 0:
                continue

            # Prevent runaway accumulation caused by repeated BC guesses.
            current_seeds = int(seeds.get(crop, 0))

            if current_seeds >= 20:
                continue

            quantity = min(
                quantity,
                20 - current_seeds,
            )

            spec = CROP_SPECS.get(crop)
            if spec is None:
                continue
            price = spec.seed_cost

            affordable = max(
                0,
                (budget - CASH_RESERVE) // price,
            )

            quantity = min(quantity, affordable)

            if quantity <= 0:
                continue

            accepted.append(
                ["BUY_SEED", crop, quantity]
            )

            budget -= price * quantity
            continue

        # ------------------------------------------------------------
        # BUY_ANIMAL
        # ------------------------------------------------------------

        if op == "BUY_ANIMAL":
            if len(order) < 3:
                continue

            animal = order[1]
            quantity = _positive_int(order[2])

            if quantity <= 0:
                continue

            # Do not allow large accidental BC purchases.
            quantity = min(quantity, 1)

            spec = ANIMAL_SPECS.get(animal)
            if spec is None:
                continue
            price = spec.cost

            affordable = max(
                0,
                (budget - CASH_RESERVE) // price,
            )

            quantity = min(quantity, affordable)

            if quantity <= 0:
                continue

            accepted.append(
                ["BUY_ANIMAL", animal, quantity]
            )

            budget -= price * quantity
            continue

        # ------------------------------------------------------------
        # SELL
        # ------------------------------------------------------------

        if op == "SELL":
            if len(order) < 3:
                continue

            item = order[1]
            quantity = _positive_int(order[2])

            if quantity <= 0:
                continue

            available = int(shed.get(item, 0))

            quantity = min(quantity, available)

            if quantity <= 0:
                continue

            accepted.append(
                ["SELL", item, quantity]
            )
            continue

    return accepted


def _positive_int(value):
    try:
        return max(0, int(round(float(value))))
    except (TypeError, ValueError):
        return 0


def _fib(n):
    """Fibonacci hire cost for multiplier=1."""
    if n <= 1:
        return 1

    a, b = 1, 1

    for _ in range(2, n + 1):
        a, b = b, a + b

    return b


def _next_land_cost(unlocked):
    """
    Approximate default Kaggriculture BUY_LAND progression.

    NW is initially unlocked.
    """

    if "NE" not in unlocked:
        return 1000

    if "SW" not in unlocked:
        return 2000

    if "SE" not in unlocked:
        return 4000

    return None


def _market_price(obs, item):
    """
    Read a price when it is available in the observation.

    This is the dynamic product price used by SELL and BUY_PRODUCT.
    Seeds and animals have fixed prices and are handled from game_data.
    """

    market = obs.get("market", {})
    prices = market.get("prices", {})

    value = prices.get(item)

    if value is None:
        return None

    try:
        value = int(value)
    except (TypeError, ValueError):
        return None

    if value <= 0:
        return None

    return value
