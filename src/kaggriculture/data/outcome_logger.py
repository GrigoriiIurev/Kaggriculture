"""Build a compact learning dataset and episode report from replay JSON files."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.kaggriculture.core.game_data import ANIMAL_SPECS, CROP_SPECS
from src.kaggriculture.data.replay_parser import Replay, infer_team_name, iter_transitions, load_replay, seats_for_team


SUMMARY_FIELDS = (
    "episode_id",
    "episode_type",
    "source_file",
    "seat",
    "team",
    "opponent",
    "result",
    "reward",
    "opponent_reward",
    "margin",
    "steps",
    "status",
    "final_money",
    "opponent_final_money",
    "unlocked_land_count",
    "final_animals_total",
    "final_cows",
    "final_sheep",
    "final_geese",
    "final_plants_total",
    "final_shed_items",
    "final_carried_items",
    "unused_seed_units",
    "unused_seed_cost",
    "bought_seed_units",
    "bought_seed_cost",
    "bought_animal_units",
    "bought_animal_cost",
    "estimated_lost_animals",
    "sold_units",
    "bought_product_units",
    "hire_orders",
    "land_orders",
    "field_action_counts",
    "market_action_counts",
    "final_seed_counts",
    "final_shed_counts",
    "final_carried_counts",
    "final_plant_counts",
)


def _json_cell(value: Mapping[str, int]) -> str:
    return json.dumps(dict(sorted(value.items())), separators=(",", ":"), sort_keys=True)


def _count_inventory(inventories: Iterable[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for inventory in inventories:
        counts.update({str(item): int(amount) for item, amount in inventory.items()})
    return counts


def _field_assets(farm: Mapping[str, Any]) -> tuple[Counter[str], Counter[str]]:
    animals: Counter[str] = Counter()
    plants: Counter[str] = Counter()
    for row in farm.get("tiles", []):
        for tile in row:
            if not isinstance(tile, Mapping):
                continue
            if tile.get("kind") == "PLANT" and tile.get("crop"):
                plants[str(tile["crop"])] += 1
            if tile.get("animal"):
                animals[str(tile["animal"])] += 1
    return animals, plants


def _owned_animals(observation: Mapping[str, Any], seat: int) -> Counter[str]:
    private = observation["private"]
    shed = _count_inventory([private.get("shed", {})])
    carried = _count_inventory(private.get("inventories", []))
    field_animals, _ = _field_assets(observation["farms"][seat])
    return field_animals + Counter(
        {animal: shed[animal] + carried[animal] for animal in ANIMAL_SPECS}
    )


def _action_counts(replay: Replay, seat: int) -> tuple[Counter[str], Counter[str], Counter[str]]:
    field_counts: Counter[str] = Counter()
    market_counts: Counter[str] = Counter()
    market_units: Counter[str] = Counter()
    for frame in replay.steps[1:]:
        action = frame[seat].get("action", {})
        if not isinstance(action, Mapping):
            continue
        field_actions = [action.get("farmer"), *action.get("hands", [])]
        for operation in field_actions:
            if isinstance(operation, list) and operation:
                field_counts[str(operation[0])] += 1
        for order in action.get("market", []):
            if not isinstance(order, list) or not order:
                continue
            operation = str(order[0])
            market_counts[operation] += 1
            amount = order[2] if len(order) > 2 and isinstance(order[2], (int, float)) else 1
            market_units[operation] += int(amount)
    return field_counts, market_counts, market_units


def summarize_trajectory(replay: Replay, seat: int) -> dict[str, Any]:
    """Create one human-readable summary row for one agent trajectory."""

    opponent_seat = 1 - seat
    final_observation = replay.steps[-1][seat]["observation"]
    farm = final_observation["farms"][seat]
    opponent_farm = final_observation["farms"][opponent_seat]
    private = final_observation["private"]
    shed = _count_inventory([private.get("shed", {})])
    seeds = _count_inventory([private.get("seeds", {})])
    carried = _count_inventory(private.get("inventories", []))
    field_animals, field_plants = _field_assets(farm)
    owned_animals = _owned_animals(final_observation, seat)
    initial_observation = replay.steps[0][seat]["observation"]
    initial_animals = _owned_animals(initial_observation, seat)
    field_actions, market_actions, market_units = _action_counts(replay, seat)

    bought_animals: Counter[str] = Counter()
    bought_seeds: Counter[str] = Counter()
    for frame in replay.steps[1:]:
        action = frame[seat].get("action", {})
        for order in action.get("market", []) if isinstance(action, Mapping) else []:
            if not isinstance(order, list) or len(order) < 2:
                continue
            amount = int(order[2]) if len(order) > 2 else 1
            if order[0] == "BUY_ANIMAL":
                bought_animals[str(order[1])] += amount
            elif order[0] == "BUY_SEED":
                bought_seeds[str(order[1])] += amount

    bought_animal_units = sum(bought_animals.values())
    bought_seed_units = sum(bought_seeds.values())
    lost_animals = sum(
        max(0, initial_animals[animal] + bought_animals[animal] - owned_animals[animal])
        for animal in ANIMAL_SPECS
    )
    reward = replay.rewards[seat]
    opponent_reward = replay.rewards[opponent_seat]
    margin = reward - opponent_reward
    result = "win" if margin > 0 else "loss" if margin < 0 else "tie"

    return {
        "episode_id": replay.episode_id,
        "episode_type": "validation" if replay.is_self_play else "public",
        "source_file": replay.source_path.name,
        "seat": seat,
        "team": replay.teams[seat],
        "opponent": replay.teams[opponent_seat],
        "result": result,
        "reward": reward,
        "opponent_reward": opponent_reward,
        "margin": margin,
        "steps": len(replay.steps) - 1,
        "status": replay.statuses[seat],
        "final_money": farm.get("money", 0),
        "opponent_final_money": opponent_farm.get("money", 0),
        "unlocked_land_count": len(farm.get("unlocked_quadrants", [])),
        "final_animals_total": sum(owned_animals.values()),
        "final_cows": owned_animals["COW"],
        "final_sheep": owned_animals["SHEEP"],
        "final_geese": owned_animals["GOOSE"],
        "final_plants_total": sum(field_plants.values()),
        "final_shed_items": sum(shed.values()),
        "final_carried_items": sum(carried.values()),
        "unused_seed_units": sum(seeds.values()),
        "unused_seed_cost": sum(
            seeds[crop] * spec.seed_cost for crop, spec in CROP_SPECS.items()
        ),
        "bought_seed_units": bought_seed_units,
        "bought_seed_cost": sum(
            bought_seeds[crop] * spec.seed_cost for crop, spec in CROP_SPECS.items()
        ),
        "bought_animal_units": bought_animal_units,
        "bought_animal_cost": sum(
            bought_animals[animal] * spec.cost for animal, spec in ANIMAL_SPECS.items()
        ),
        "estimated_lost_animals": lost_animals,
        "sold_units": market_units["SELL"],
        "bought_product_units": market_units["BUY_PRODUCT"],
        "hire_orders": market_actions["HIRE"],
        "land_orders": market_actions["BUY_LAND"],
        "field_action_counts": _json_cell(field_actions),
        "market_action_counts": _json_cell(market_actions),
        "final_seed_counts": _json_cell(seeds),
        "final_shed_counts": _json_cell(shed),
        "final_carried_counts": _json_cell(carried),
        "final_plant_counts": _json_cell(field_plants),
    }


def build_dataset(
    replay_directory: str | Path,
    output_directory: str | Path,
    team_name: str | None = None,
) -> dict[str, Any]:
    """Convert all unique JSON replays in a directory into three dataset files."""

    replay_directory = Path(replay_directory)
    output_directory = Path(output_directory)
    paths = sorted(replay_directory.glob("*.json"))
    if not paths:
        raise ValueError(f"No JSON replay files found in {replay_directory}")
    team_name = team_name or infer_team_name(paths)
    output_directory.mkdir(parents=True, exist_ok=True)

    transitions_path = output_directory / "transitions.jsonl.gz"
    episodes_path = output_directory / "episodes.csv"
    manifest_path = output_directory / "manifest.json"
    seen_episode_ids: set[int] = set()
    skipped_duplicates: list[str] = []
    summaries: list[dict[str, Any]] = []
    transition_count = 0
    public_episode_count = 0
    validation_episode_count = 0

    with gzip.open(transitions_path, "wt", encoding="utf-8", newline="\n") as output:
        for path in paths:
            replay = load_replay(path)
            if replay.episode_id in seen_episode_ids:
                skipped_duplicates.append(path.name)
                continue
            seen_episode_ids.add(replay.episode_id)
            seats = seats_for_team(replay, team_name)
            if not seats:
                continue
            if replay.is_self_play:
                validation_episode_count += 1
            else:
                public_episode_count += 1
            for seat in seats:
                summaries.append(summarize_trajectory(replay, seat))
            for transition in iter_transitions(replay, team_name):
                json.dump(transition.as_dict(), output, separators=(",", ":"))
                output.write("\n")
                transition_count += 1

    with episodes_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    public_rows = [row for row in summaries if row["episode_type"] == "public"]
    result_counts = Counter(str(row["result"]) for row in public_rows)
    manifest = {
        "team_name": team_name,
        "source_directory": str(replay_directory.resolve()),
        "json_files_found": len(paths),
        "unique_episodes_used": public_episode_count + validation_episode_count,
        "public_episodes": public_episode_count,
        "validation_episodes": validation_episode_count,
        "trajectories": len(summaries),
        "transitions": transition_count,
        "public_results": {
            "wins": result_counts["win"],
            "losses": result_counts["loss"],
            "ties": result_counts["tie"],
        },
        "skipped_duplicate_files": skipped_duplicates,
        "files": {
            "transitions": transitions_path.name,
            "episodes": episodes_path.name,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=True, indent=2)
        output.write("\n")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_directory", nargs="?", default="replays")
    parser.add_argument("--output", default="replay_dataset", dest="output_directory")
    parser.add_argument("--team", help="Team name; inferred from replay frequency when omitted")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_dataset(args.replay_directory, args.output_directory, args.team)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
