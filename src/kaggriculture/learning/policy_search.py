"""Learn economic planner parameters by measuring complete game outcomes."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from kaggle_environments import make

from ..agent import RuleBasedAgent
from ..planning.economic_planner import EconomicConfig


@dataclass(frozen=True)
class MatchResult:
    seed: int
    candidate_seat: int
    candidate_reward: float
    baseline_reward: float

    @property
    def margin(self) -> float:
        return self.candidate_reward - self.baseline_reward


def candidate_configs() -> dict[str, EconomicConfig]:
    """Small, interpretable search space around the proven baseline."""

    base = EconomicConfig()
    return {
        "baseline": base,
        "lower_reserve": replace(base, cash_reserve=100),
        "higher_reserve": replace(base, cash_reserve=500),
        "faster_expansion": replace(base, first_land_day=4, second_land_day=7),
        "later_expansion": replace(base, first_land_day=7, second_land_day=12),
        "cow_heavy": replace(
            base,
            bootstrap_cows=2,
            bootstrap_sheep=3,
            growth_cows=6,
            growth_sheep=2,
            mature_cows=10,
            mature_sheep=2,
        ),
        "sheep_heavy": replace(
            base,
            bootstrap_cows=0,
            bootstrap_sheep=5,
            growth_cows=2,
            growth_sheep=6,
            mature_cows=5,
            mature_sheep=7,
        ),
        "lean_livestock": replace(
            base,
            bootstrap_cows=1,
            bootstrap_sheep=2,
            growth_cows=3,
            growth_sheep=3,
            mature_cows=6,
            mature_sheep=3,
        ),
        "larger_batches": replace(base, seed_batch_per_crop=8, feed_days=4),
        "fast_inventory_return": replace(base, inventory_return_threshold=3),
    }


def play_match(config: EconomicConfig, seed: int, candidate_seat: int) -> MatchResult:
    candidate = RuleBasedAgent(config)
    baseline = RuleBasedAgent(EconomicConfig())
    agents = [candidate, baseline] if candidate_seat == 0 else [baseline, candidate]
    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": seed},
        debug=False,
    )
    environment.run(agents)
    final = environment.steps[-1]
    statuses = [str(player.status) for player in final]
    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(
            f"Local game did not complete: statuses={statuses}, seed={seed}"
        )
    rewards = [float(player.reward or 0.0) for player in final]
    candidate_reward = rewards[candidate_seat]
    baseline_reward = rewards[1 - candidate_seat]
    return MatchResult(seed, candidate_seat, candidate_reward, baseline_reward)


def evaluate_config(
    config: EconomicConfig,
    games: int,
    seed_offset: int,
) -> dict:
    if games < 2:
        raise ValueError("At least two games are required")

    results = [
        play_match(config, seed_offset + index // 2, index % 2)
        for index in range(games)
    ]
    margins = [result.margin for result in results]
    wins = sum(margin > 0 for margin in margins)
    losses = sum(margin < 0 for margin in margins)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "ties": games - wins - losses,
        "win_rate": wins / games,
        "mean_margin": statistics.mean(margins),
        "median_margin": statistics.median(margins),
        "mean_candidate_reward": statistics.mean(
            result.candidate_reward for result in results
        ),
        "mean_baseline_reward": statistics.mean(
            result.baseline_reward for result in results
        ),
        "matches": [
            {
                **asdict(result),
                "margin": result.margin,
            }
            for result in results
        ],
    }


def run_search(
    screening_games: int = 2,
    confirmation_games: int = 8,
    validation_games: int = 8,
    finalists: int = 2,
    report_path: str | Path = "artifacts/reports/policy_search.json",
    promoted_path: str | Path = "artifacts/models/promoted_economic_config.json",
) -> dict:
    configurations = candidate_configs()
    screening: dict[str, dict] = {}
    started = time.perf_counter()

    for index, (name, config) in enumerate(configurations.items()):
        if name == "baseline":
            continue
        print(f"Screening {name} ({index}/{len(configurations) - 1})...", flush=True)
        screening[name] = evaluate_config(config, screening_games, 10_000)

    ordered = sorted(
        screening,
        key=lambda name: (
            screening[name]["mean_margin"],
            screening[name]["win_rate"],
        ),
        reverse=True,
    )
    selected = ordered[: max(1, finalists)]
    confirmation: dict[str, dict] = {}
    for name in selected:
        print(f"Confirming {name}...", flush=True)
        confirmation[name] = evaluate_config(
            configurations[name], confirmation_games, 20_000
        )

    winner = max(
        selected,
        key=lambda name: (
            confirmation[name]["mean_margin"],
            confirmation[name]["win_rate"],
        ),
    )
    print(f"Validating {winner} on unseen seeds...", flush=True)
    final_validation = evaluate_config(
        configurations[winner], validation_games, 30_000
    )
    winner_result = final_validation
    promoted = (
        winner_result["wins"] > winner_result["losses"]
        and winner_result["mean_margin"] > 0
    )

    report = {
        "method": "paired_full_game_policy_search",
        "screening_games": screening_games,
        "confirmation_games": confirmation_games,
        "validation_games": validation_games,
        "screening": screening,
        "confirmation": confirmation,
        "winner": winner,
        "final_validation": final_validation,
        "promoted": promoted,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    if promoted:
        promoted_path = Path(promoted_path)
        promoted_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": winner,
            "config": asdict(configurations[winner]),
            "validation": winner_result,
        }
        with promoted_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    return report
