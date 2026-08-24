"""Build compact analytical datasets from Kaggriculture replay files."""

from __future__ import annotations

import csv
import gzip
import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

from src.kaggriculture.core.game_data import ANIMAL_SPECS, CROP_SPECS, PRODUCTS
from src.kaggriculture.data.replay_parser import (
    Replay,
    ReplayParseError,
    infer_team_name,
    load_replay,
    normalize_observation,
    seats_for_team,
)


ANIMALS = tuple(ANIMAL_SPECS)
CROPS = tuple(CROP_SPECS)
PREMIUM_CROPS = ("STRAWBERRY", "MELON")
TRAJECTORY_FIELDS = (
    "episode_id",
    "episode_type",
    "seat",
    "team",
    "opponent",
    "result",
    "reward",
    "opponent_reward",
    "margin",
    "strategy",
    "peak_hands",
    "peak_land",
    "peak_animals",
    "peak_cows",
    "peak_sheep",
    "peak_geese",
    "peak_plants",
    "peak_premium_plants",
    "first_animal_day",
    "first_land_day",
    "hire_orders",
    "land_orders",
    "sell_units",
    "buy_product_units",
    "bought_animals",
    "bought_seeds",
    "sold_products",
)
EPISODE_FIELDS = (
    "episode_id",
    "episode_type",
    "created_time",
    "team",
    "opponent",
    "seat",
    "result",
    "reward",
    "opponent_reward",
    "margin",
    "opponent_strategy",
    "deficit_phase",
    "diagnostic_tag",
    "day5_margin",
    "day10_margin",
    "day15_margin",
    "day20_margin",
    "day25_margin",
    "worst_money_margin",
    "best_money_margin",
    "our_peak_animals",
    "opponent_peak_animals",
    "our_peak_hands",
    "opponent_peak_hands",
    "our_peak_land",
    "opponent_peak_land",
)


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _field_counts(farm: Mapping[str, Any]) -> dict[str, Any]:
    animals: Counter[str] = Counter()
    plants: Counter[str] = Counter()
    weeds = structures = yield_units = 0
    for row in farm.get("tiles", []):
        for tile in row:
            if not isinstance(tile, Mapping):
                continue
            kind = str(tile.get("kind", ""))
            if kind == "PLANT" and tile.get("crop"):
                plants[str(tile["crop"])] += 1
            if tile.get("animal"):
                animals[str(tile["animal"])] += 1
            if kind == "WEED":
                weeds += 1
            if kind in {"COOP", "PASTURE"}:
                structures += 1
            yield_units += max(0, int(tile.get("yield_units", 0) or 0))
    return {
        "money": float(farm.get("money", 0)),
        "hands": len(farm.get("hands", [])),
        "hires_today": int(farm.get("hires_today", 0)),
        "land": len(farm.get("unlocked_quadrants", [])),
        "animals": {animal: animals[animal] for animal in ANIMALS},
        "plants": {crop: plants[crop] for crop in CROPS},
        "animal_total": sum(animals.values()),
        "plant_total": sum(plants.values()),
        "premium_plant_total": sum(plants[crop] for crop in PREMIUM_CROPS),
        "weeds": weeds,
        "structures": structures,
        "yield_units": yield_units,
    }


def _private_counts(observation: Mapping[str, Any]) -> dict[str, Any]:
    private = observation.get("private", {})
    carried: Counter[str] = Counter()
    for inventory in private.get("inventories", []):
        if isinstance(inventory, Mapping):
            carried.update({str(k): int(v) for k, v in inventory.items()})
    return {
        "shed": {str(k): int(v) for k, v in private.get("shed", {}).items()},
        "seeds": {str(k): int(v) for k, v in private.get("seeds", {}).items()},
        "carried": dict(carried),
    }


def _compact_state(observation: Mapping[str, Any], seat: int) -> dict[str, Any]:
    opponent = 1 - seat
    return {
        "step": int(observation["step"]),
        "day": int(observation["day"]),
        "hour": int(observation["hour"]),
        "me": _field_counts(observation["farms"][seat]),
        "opponent": _field_counts(observation["farms"][opponent]),
        "private": _private_counts(observation),
        "market_prices": {
            item: int(observation.get("market", {}).get("prices", {}).get(item, 0))
            for item in PRODUCTS
        },
        "market_inventory": {
            item: int(observation.get("market", {}).get("inventory", {}).get(item, 0))
            for item in PRODUCTS
        },
        "shops": list(observation.get("town", {}).get("unlocked_shops", [])),
    }


def _update_peak(profile: dict[str, Any], farm: Mapping[str, Any], day: int) -> None:
    counts = _field_counts(farm)
    profile["peak_hands"] = max(profile["peak_hands"], counts["hands"])
    profile["peak_land"] = max(profile["peak_land"], counts["land"])
    profile["peak_animals"] = max(profile["peak_animals"], counts["animal_total"])
    profile["peak_plants"] = max(profile["peak_plants"], counts["plant_total"])
    profile["peak_premium_plants"] = max(
        profile["peak_premium_plants"], counts["premium_plant_total"]
    )
    animal_peak_keys = {
        "COW": "peak_cows",
        "SHEEP": "peak_sheep",
        "GOOSE": "peak_geese",
    }
    for animal, key in animal_peak_keys.items():
        profile[key] = max(profile[key], counts["animals"][animal])
    if counts["animal_total"] and profile["first_animal_day"] is None:
        profile["first_animal_day"] = day
    if counts["land"] > 1 and profile["first_land_day"] is None:
        profile["first_land_day"] = day


def _new_profile(replay: Replay, seat: int) -> dict[str, Any]:
    opponent = 1 - seat
    reward = replay.rewards[seat]
    opponent_reward = replay.rewards[opponent]
    margin = reward - opponent_reward
    return {
        "episode_id": replay.episode_id,
        "episode_type": "validation" if replay.is_self_play else "public",
        "seat": seat,
        "team": replay.teams[seat],
        "opponent": replay.teams[opponent],
        "result": "win" if margin > 0 else "loss" if margin < 0 else "tie",
        "reward": reward,
        "opponent_reward": opponent_reward,
        "margin": margin,
        "strategy": "unknown",
        "peak_hands": 0,
        "peak_land": 0,
        "peak_animals": 0,
        "peak_cows": 0,
        "peak_sheep": 0,
        "peak_geese": 0,
        "peak_plants": 0,
        "peak_premium_plants": 0,
        "first_animal_day": None,
        "first_land_day": None,
        "hire_orders": 0,
        "land_orders": 0,
        "sell_units": 0,
        "buy_product_units": 0,
        "bought_animals": Counter(),
        "bought_seeds": Counter(),
        "sold_products": Counter(),
    }


def _record_action(profile: dict[str, Any], action: Mapping[str, Any]) -> None:
    for order in action.get("market", []):
        if not isinstance(order, list) or not order:
            continue
        operation = str(order[0])
        item = str(order[1]) if len(order) > 1 else ""
        quantity = int(order[2]) if len(order) > 2 else 1
        if operation == "HIRE":
            profile["hire_orders"] += 1
        elif operation == "BUY_LAND":
            profile["land_orders"] += 1
        elif operation == "BUY_ANIMAL":
            profile["bought_animals"][item] += quantity
        elif operation == "BUY_SEED":
            profile["bought_seeds"][item] += quantity
        elif operation == "BUY_PRODUCT":
            profile["buy_product_units"] += quantity
        elif operation == "SELL":
            profile["sell_units"] += quantity
            profile["sold_products"][item] += quantity


def _classify_strategy(profile: Mapping[str, Any]) -> str:
    cows = int(profile["peak_cows"])
    sheep = int(profile["peak_sheep"])
    geese = int(profile["peak_geese"])
    animals = int(profile["peak_animals"])
    premium = int(profile["peak_premium_plants"])
    plants = int(profile["peak_plants"])
    if cows >= 7 and cows >= sheep * 1.5:
        return "cow_heavy"
    if sheep >= 7 and sheep >= cows * 1.5:
        return "sheep_heavy"
    if geese >= 7 and geese >= max(cows, sheep):
        return "goose_heavy"
    if animals >= 8 and premium >= 5:
        return "mixed_farm"
    if animals >= 8:
        return "mixed_livestock"
    if premium >= 8:
        return "premium_crops"
    if plants >= 10:
        return "crop_focused"
    if int(profile["peak_land"]) >= 4:
        return "expansion_low_density"
    return "low_scale_or_other"


def _finalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    profile["strategy"] = _classify_strategy(profile)
    for key in ("bought_animals", "bought_seeds", "sold_products"):
        profile[key] = _json_cell(dict(sorted(profile[key].items())))
    return {field: profile.get(field) for field in TRAJECTORY_FIELDS}


def _action_summary(action: Mapping[str, Any]) -> dict[str, Any]:
    market = [list(order) for order in action.get("market", []) if isinstance(order, list)]
    field = []
    farmer = action.get("farmer")
    if isinstance(farmer, list):
        field.append(farmer)
    field.extend(order for order in action.get("hands", []) if isinstance(order, list))
    return {"field": field, "market": market}


def _write_jsonl(output: TextIO, record: Mapping[str, Any]) -> None:
    json.dump(record, output, ensure_ascii=False, separators=(",", ":"))
    output.write("\n")


def _flush_macro_day(
    output: TextIO,
    replay: Replay,
    seat: int,
    accumulator: dict[str, Any],
) -> None:
    if not accumulator:
        return
    opponent = 1 - seat
    margin = replay.rewards[seat] - replay.rewards[opponent]
    record = {
        "episode_id": replay.episode_id,
        "episode_type": "validation" if replay.is_self_play else "public",
        "seat": seat,
        "team": replay.teams[seat],
        "opponent": replay.teams[opponent],
        "day": accumulator["day"],
        "result": "win" if margin > 0 else "loss" if margin < 0 else "tie",
        "is_winner": margin > 0,
        "final_margin": margin,
        "start": accumulator["start"],
        "end": accumulator["end"],
        "money_delta": accumulator["end"]["me"]["money"]
        - accumulator["start"]["me"]["money"],
        "field_actions": dict(accumulator["field_actions"]),
        "market_operations": dict(accumulator["market_operations"]),
        "market_units": dict(accumulator["market_units"]),
    }
    _write_jsonl(output, record)


def _day_margin(margins: Mapping[int, float], day: int) -> float | None:
    eligible = [value for key, value in margins.items() if key <= day]
    return eligible[-1] if eligible else None


def _deficit_phase(result: str, margins: Mapping[int, float]) -> str:
    if result != "loss":
        return "not_a_loss"
    for day, label in ((5, "early"), (15, "midgame"), (25, "late")):
        value = _day_margin(margins, day)
        if value is not None and value <= -1000:
            return label
    return "finish"


def _diagnostic_tag(
    result: str,
    phase: str,
    ours: Mapping[str, Any],
    opponent: Mapping[str, Any],
) -> str:
    if result != "loss":
        return "not_a_loss"
    if int(opponent["peak_animals"]) >= int(ours["peak_animals"]) + 3:
        return "opponent_production_scale"
    if int(opponent["peak_premium_plants"]) >= int(ours["peak_premium_plants"]) + 4:
        return "opponent_premium_crop_scale"
    if phase in {"late", "finish"}:
        return "late_market_or_liquidation"
    if int(opponent["peak_land"]) > int(ours["peak_land"]):
        return "opponent_expansion"
    return "execution_or_market_timing"


def _episode_row(
    replay: Replay,
    seat: int,
    ours: Mapping[str, Any],
    opponent: Mapping[str, Any],
    margins: Mapping[int, float],
    created_time: str | None,
) -> dict[str, Any]:
    result = str(ours["result"])
    phase = _deficit_phase(result, margins)
    values = list(margins.values()) or [float(ours["margin"])]
    return {
        "episode_id": replay.episode_id,
        "episode_type": ours["episode_type"],
        "created_time": created_time or "",
        "team": ours["team"],
        "opponent": ours["opponent"],
        "seat": seat,
        "result": result,
        "reward": ours["reward"],
        "opponent_reward": ours["opponent_reward"],
        "margin": ours["margin"],
        "opponent_strategy": opponent["strategy"],
        "deficit_phase": phase,
        "diagnostic_tag": _diagnostic_tag(result, phase, ours, opponent),
        "day5_margin": _day_margin(margins, 5),
        "day10_margin": _day_margin(margins, 10),
        "day15_margin": _day_margin(margins, 15),
        "day20_margin": _day_margin(margins, 20),
        "day25_margin": _day_margin(margins, 25),
        "worst_money_margin": min(values),
        "best_money_margin": max(values),
        "our_peak_animals": ours["peak_animals"],
        "opponent_peak_animals": opponent["peak_animals"],
        "our_peak_hands": ours["peak_hands"],
        "opponent_peak_hands": opponent["peak_hands"],
        "our_peak_land": ours["peak_land"],
        "opponent_peak_land": opponent["peak_land"],
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    public = [row for row in rows if row["episode_type"] == "public"]
    outcomes = Counter(str(row["result"]) for row in public)
    margins = [float(row["margin"]) for row in public]

    def grouped(field: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in public:
            groups[str(row[field])].append(row)
        return {
            name: {
                "games": len(items),
                "wins": sum(item["result"] == "win" for item in items),
                "ties": sum(item["result"] == "tie" for item in items),
                "losses": sum(item["result"] == "loss" for item in items),
                "win_rate": sum(item["result"] == "win" for item in items)
                / len(items),
                "mean_margin": statistics.fmean(float(item["margin"]) for item in items),
            }
            for name, items in sorted(groups.items())
        }

    return {
        "games": len(public),
        "wins": outcomes["win"],
        "ties": outcomes["tie"],
        "losses": outcomes["loss"],
        "win_rate": outcomes["win"] / len(public) if public else 0.0,
        "score_rate": (outcomes["win"] + 0.5 * outcomes["tie"]) / len(public)
        if public
        else 0.0,
        "mean_margin": statistics.fmean(margins) if margins else 0.0,
        "median_margin": statistics.median(margins) if margins else 0.0,
        "by_seat": grouped("seat"),
        "by_opponent_strategy": grouped("opponent_strategy"),
        "loss_deficit_phase": dict(
            Counter(
                str(row["deficit_phase"])
                for row in public
                if row["result"] == "loss"
            )
        ),
        "loss_diagnostic_tags": dict(
            Counter(
                str(row["diagnostic_tag"])
                for row in public
                if row["result"] == "loss"
            )
        ),
        "worst_losses": sorted(
            (row for row in public if row["result"] == "loss"),
            key=lambda row: float(row["margin"]),
        )[:10],
        "best_wins": sorted(
            (row for row in public if row["result"] == "win"),
            key=lambda row: float(row["margin"]),
            reverse=True,
        )[:10],
    }


def _markdown_report(manifest: Mapping[str, Any]) -> str:
    summary = manifest["summary"]
    lines = [
        "# Kaggriculture Replay Warehouse Report",
        "",
        f"- Submission: `{manifest.get('submission_id', 'unknown')}`",
        f"- Team: `{manifest['team_name']}`",
        f"- Public games: **{summary['games']}**",
        f"- Record: **{summary['wins']}W-{summary['ties']}T-{summary['losses']}L**",
        f"- Win rate: **{summary['win_rate']:.1%}**",
        f"- Score rate: **{summary['score_rate']:.1%}**",
        f"- Mean margin: **{summary['mean_margin']:+.1f}**",
        f"- Median margin: **{summary['median_margin']:+.1f}**",
        "",
        "## By Seat",
        "",
        "| Seat | Games | W-T-L | Win rate | Mean margin |",
        "|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary["by_seat"].items():
        lines.append(
            f"| {name} | {item['games']} | {item['wins']}-{item['ties']}-{item['losses']} "
            f"| {item['win_rate']:.1%} | {item['mean_margin']:+.1f} |"
        )
    lines.extend(
        [
            "",
            "## Opponent Strategies",
            "",
            "| Strategy | Games | W-T-L | Win rate | Mean margin |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in summary["by_opponent_strategy"].items():
        lines.append(
            f"| {name} | {item['games']} | {item['wins']}-{item['ties']}-{item['losses']} "
            f"| {item['win_rate']:.1%} | {item['mean_margin']:+.1f} |"
        )
    lines.extend(["", "## Loss Diagnostics", ""])
    for name, count in sorted(summary["loss_diagnostic_tags"].items()):
        lines.append(f"- `{name}`: {count}")
    lines.extend(
        [
            "",
            "These diagnostic tags are heuristics. They identify where to inspect a replay; "
            "they do not prove causality.",
            "",
            "## Worst Losses",
            "",
            "| Episode | Opponent | Strategy | Margin | Phase | Diagnostic |",
            "|---:|---|---|---:|---|---|",
        ]
    )
    for row in summary["worst_losses"]:
        lines.append(
            f"| {row['episode_id']} | {row['opponent']} | {row['opponent_strategy']} "
            f"| {float(row['margin']):+.0f} | {row['deficit_phase']} "
            f"| {row['diagnostic_tag']} |"
        )
    lines.extend(
        [
            "",
            "## Dataset Files",
            "",
            f"- `episodes.csv`: {manifest['rows']['episodes']:,} submitted-agent trajectories",
            f"- `trajectory_profiles.csv`: {manifest['rows']['trajectory_profiles']:,} player profiles",
            f"- `daily_macro.jsonl.gz`: {manifest['rows']['daily_macro']:,} player-days",
            f"- `market_decisions.jsonl.gz`: {manifest['rows']['market_decisions']:,} player-turns",
            "",
        ]
    )
    return "\n".join(lines)


def _episode_times(
    index_path: str | Path | None,
) -> tuple[dict[int, str], int | None, set[int] | None]:
    if index_path is None or not Path(index_path).is_file():
        return {}, None, None
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    times = {
        int(item["id"]): str(item.get("createTime", ""))
        for item in payload.get("episodes", [])
        if "id" in item
    }
    submission_id = payload.get("submission", {}).get("ref")
    return (
        times,
        int(submission_id) if submission_id is not None else None,
        set(times),
    )


def build_replay_warehouse(
    replay_directory: str | Path,
    output_directory: str | Path,
    team_name: str | None = None,
    episode_index_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build compact episode, daily, and market datasets plus a report."""

    replay_directory = Path(replay_directory)
    output_directory = Path(output_directory)
    paths = sorted(replay_directory.glob("*.json"))
    if not paths:
        raise ValueError(f"No replay JSON files found in {replay_directory}")
    episode_times, submission_id, selected_ids = _episode_times(episode_index_path)
    if selected_ids is not None:
        selected_paths = []
        for path in paths:
            match = re.search(r"\d{6,}", path.stem)
            if match:
                episode_id = int(match.group())
            else:
                try:
                    episode_id = load_replay(path).episode_id
                except ReplayParseError:
                    selected_paths.append(path)
                    continue
            if episode_id in selected_ids:
                selected_paths.append(path)
        paths = selected_paths
    if not paths:
        raise ValueError("No replay files match episode_index.json")
    team_name = team_name or infer_team_name(paths)
    output_directory.mkdir(parents=True, exist_ok=True)

    profiles_path = output_directory / "trajectory_profiles.csv"
    episodes_path = output_directory / "episodes.csv"
    macro_path = output_directory / "daily_macro.jsonl.gz"
    market_path = output_directory / "market_decisions.jsonl.gz"
    manifest_path = output_directory / "manifest.json"
    report_path = output_directory / "report.md"

    seen: set[int] = set()
    skipped: list[dict[str, str]] = []
    profile_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    macro_count = market_count = 0

    with gzip.open(macro_path, "wt", encoding="utf-8") as macro_output, gzip.open(
        market_path, "wt", encoding="utf-8"
    ) as market_output:
        for number, path in enumerate(paths, start=1):
            if number == 1 or number % 10 == 0 or number == len(paths):
                print(
                    f"[warehouse] replay {number:,}/{len(paths):,}: {path.name}; "
                    f"{market_count:,} market rows",
                    flush=True,
                )
            try:
                replay = load_replay(path)
            except ReplayParseError as exc:
                skipped.append({"file": path.name, "reason": str(exc)})
                continue
            if replay.episode_id in seen:
                skipped.append({"file": path.name, "reason": "duplicate_episode"})
                continue
            seen.add(replay.episode_id)
            profiles = [_new_profile(replay, seat) for seat in range(2)]
            daily_margins = [dict(), dict()]
            day_accumulators: list[dict[str, Any]] = [{}, {}]

            for frame_index, frame in enumerate(replay.steps):
                primary_observation = frame[0].get("observation", {})
                day = int(primary_observation.get("day", frame_index // replay.turns_per_day))
                hour = int(primary_observation.get("hour", frame_index % replay.turns_per_day))
                farms = primary_observation.get("farms", [])
                if len(farms) == 2:
                    for seat in range(2):
                        _update_peak(profiles[seat], farms[seat], day)
                    if hour == replay.turns_per_day - 1 or frame_index == len(replay.steps) - 1:
                        margin0 = float(farms[0].get("money", 0)) - float(
                            farms[1].get("money", 0)
                        )
                        daily_margins[0][day] = margin0
                        daily_margins[1][day] = -margin0

                if frame_index >= len(replay.steps) - 1:
                    continue
                following = replay.steps[frame_index + 1]
                for seat in range(2):
                    raw_observation = frame[seat].get("observation")
                    raw_next = following[seat].get("observation")
                    action = following[seat].get("action")
                    if not isinstance(raw_observation, Mapping) or not isinstance(
                        raw_next, Mapping
                    ) or not isinstance(action, Mapping):
                        continue
                    observation = normalize_observation(
                        raw_observation, replay.turns_per_day, frame_index
                    )
                    next_observation = normalize_observation(
                        raw_next, replay.turns_per_day, frame_index + 1
                    )
                    state = _compact_state(observation, seat)
                    next_state = _compact_state(next_observation, seat)
                    opponent = 1 - seat
                    final_margin = replay.rewards[seat] - replay.rewards[opponent]
                    result = (
                        "win" if final_margin > 0 else "loss" if final_margin < 0 else "tie"
                    )
                    action_summary = _action_summary(action)
                    _record_action(profiles[seat], action)
                    _write_jsonl(
                        market_output,
                        {
                            "episode_id": replay.episode_id,
                            "episode_type": "validation" if replay.is_self_play else "public",
                            "seat": seat,
                            "team": replay.teams[seat],
                            "opponent": replay.teams[opponent],
                            "result": result,
                            "is_winner": final_margin > 0,
                            "final_margin": final_margin,
                            "state": state,
                            "market_orders": action_summary["market"],
                            "next_money_delta": next_state["me"]["money"]
                            - state["me"]["money"],
                        },
                    )
                    market_count += 1

                    accumulator = day_accumulators[seat]
                    if accumulator and accumulator["day"] != state["day"]:
                        _flush_macro_day(macro_output, replay, seat, accumulator)
                        macro_count += 1
                        accumulator = {}
                        day_accumulators[seat] = accumulator
                    if not accumulator:
                        accumulator.update(
                            {
                                "day": state["day"],
                                "start": state,
                                "end": next_state,
                                "field_actions": Counter(),
                                "market_operations": Counter(),
                                "market_units": Counter(),
                            }
                        )
                    accumulator["end"] = next_state
                    for command in action_summary["field"]:
                        if command:
                            accumulator["field_actions"][str(command[0])] += 1
                    for order in action_summary["market"]:
                        if not order:
                            continue
                        operation = str(order[0])
                        item = str(order[1]) if len(order) > 1 else ""
                        quantity = int(order[2]) if len(order) > 2 else 1
                        accumulator["market_operations"][operation] += 1
                        accumulator["market_units"][f"{operation}:{item}"] += quantity

            for seat, accumulator in enumerate(day_accumulators):
                if accumulator:
                    _flush_macro_day(macro_output, replay, seat, accumulator)
                    macro_count += 1

            finalized = [_finalize_profile(profile) for profile in profiles]
            profile_rows.extend(finalized)
            for seat in seats_for_team(replay, team_name):
                episode_rows.append(
                    _episode_row(
                        replay,
                        seat,
                        finalized[seat],
                        finalized[1 - seat],
                        daily_margins[seat],
                        episode_times.get(replay.episode_id),
                    )
                )

    with profiles_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=TRAJECTORY_FIELDS)
        writer.writeheader()
        writer.writerows(profile_rows)
    with episodes_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(episode_rows)

    summary = _aggregate_rows(episode_rows)
    manifest = {
        "version": 1,
        "submission_id": submission_id,
        "team_name": team_name,
        "source_directory": str(replay_directory.resolve()),
        "json_files_found": len(paths),
        "unique_episodes": len(seen),
        "skipped": skipped,
        "rows": {
            "episodes": len(episode_rows),
            "trajectory_profiles": len(profile_rows),
            "daily_macro": macro_count,
            "market_decisions": market_count,
        },
        "summary": summary,
        "files": {
            "episodes": episodes_path.name,
            "trajectory_profiles": profiles_path.name,
            "daily_macro": macro_path.name,
            "market_decisions": market_path.name,
            "report": report_path.name,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_markdown_report(manifest), encoding="utf-8")
    print(
        f"[warehouse] complete: {len(episode_rows):,} episodes, "
        f"{macro_count:,} player-days, {market_count:,} player-turns",
        flush=True,
    )
    return manifest
