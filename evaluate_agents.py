"""Evaluate Kaggriculture agents over multiple games."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from kaggle_environments import make


DEFAULT_AGENT_A = "ml_agent.py"
DEFAULT_AGENT_B = "behavior_agent.py"
DEFAULT_GAMES = 20
DEFAULT_EPISODE_STEPS = 720


@dataclass
class GameResult:
    game: int
    agent_a_player: int
    reward_a: float
    reward_b: float
    margin_a: float
    winner: str
    status_a: str
    status_b: str
    elapsed_seconds: float


def safe_reward(value) -> float:
    if value is None:
        return 0.0

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def play_game(
    agent_a: str,
    agent_b: str,
    game_number: int,
    agent_a_player: int,
    episode_steps: int,
) -> GameResult:
    if agent_a_player == 0:
        agents = [agent_a, agent_b]
    else:
        agents = [agent_b, agent_a]

    started = time.perf_counter()

    env = make(
        "kaggriculture",
        configuration={
            "episodeSteps": episode_steps,
        },
        debug=False,
    )

    env.run(agents)

    elapsed = time.perf_counter() - started

    final = env.steps[-1]

    player0_reward = safe_reward(
        final[0].reward
    )

    player1_reward = safe_reward(
        final[1].reward
    )

    if agent_a_player == 0:
        reward_a = player0_reward
        reward_b = player1_reward

        status_a = str(final[0].status)
        status_b = str(final[1].status)
    else:
        reward_a = player1_reward
        reward_b = player0_reward

        status_a = str(final[1].status)
        status_b = str(final[0].status)

    margin = reward_a - reward_b

    if margin > 0:
        winner = "A"
    elif margin < 0:
        winner = "B"
    else:
        winner = "TIE"

    return GameResult(
        game=game_number,
        agent_a_player=agent_a_player,
        reward_a=reward_a,
        reward_b=reward_b,
        margin_a=margin,
        winner=winner,
        status_a=status_a,
        status_b=status_b,
        elapsed_seconds=elapsed,
    )


def mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return statistics.mean(values)


def median(values: list[float]) -> float:
    if not values:
        return 0.0

    return statistics.median(values)


def summarize(
    results: list[GameResult],
) -> dict:
    wins_a = sum(
        result.winner == "A"
        for result in results
    )

    wins_b = sum(
        result.winner == "B"
        for result in results
    )

    ties = sum(
        result.winner == "TIE"
        for result in results
    )

    rewards_a = [
        result.reward_a
        for result in results
    ]

    rewards_b = [
        result.reward_b
        for result in results
    ]

    margins = [
        result.margin_a
        for result in results
    ]

    player0_results = [
        result
        for result in results
        if result.agent_a_player == 0
    ]

    player1_results = [
        result
        for result in results
        if result.agent_a_player == 1
    ]

    def side_summary(
        subset: list[GameResult],
    ) -> dict:
        return {
            "games": len(subset),
            "wins": sum(
                result.winner == "A"
                for result in subset
            ),
            "losses": sum(
                result.winner == "B"
                for result in subset
            ),
            "ties": sum(
                result.winner == "TIE"
                for result in subset
            ),
            "average_reward": mean(
                [
                    result.reward_a
                    for result in subset
                ]
            ),
            "average_margin": mean(
                [
                    result.margin_a
                    for result in subset
                ]
            ),
        }

    return {
        "games": len(results),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": (
            wins_a / len(results)
            if results
            else 0.0
        ),
        "average_reward_a": mean(
            rewards_a
        ),
        "average_reward_b": mean(
            rewards_b
        ),
        "median_reward_a": median(
            rewards_a
        ),
        "median_reward_b": median(
            rewards_b
        ),
        "average_margin_a": mean(
            margins
        ),
        "median_margin_a": median(
            margins
        ),
        "agent_a_as_player_0": (
            side_summary(player0_results)
        ),
        "agent_a_as_player_1": (
            side_summary(player1_results)
        ),
        "average_game_seconds": mean(
            [
                result.elapsed_seconds
                for result in results
            ]
        ),
    }


def print_summary(
    agent_a: str,
    agent_b: str,
    summary: dict,
) -> None:
    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"Agent A: {agent_a}")
    print(f"Agent B: {agent_b}")

    print()
    print(f"Games:   {summary['games']}")
    print(f"A wins:  {summary['wins_a']}")
    print(f"B wins:  {summary['wins_b']}")
    print(f"Ties:    {summary['ties']}")

    print(
        f"A win rate: "
        f"{summary['win_rate_a']:.1%}"
    )

    print()
    print(
        f"A average reward: "
        f"{summary['average_reward_a']:.2f}"
    )

    print(
        f"B average reward: "
        f"{summary['average_reward_b']:.2f}"
    )

    print(
        f"A median reward:  "
        f"{summary['median_reward_a']:.2f}"
    )

    print(
        f"B median reward:  "
        f"{summary['median_reward_b']:.2f}"
    )

    print()
    print(
        f"A average margin: "
        f"{summary['average_margin_a']:+.2f}"
    )

    print(
        f"A median margin:  "
        f"{summary['median_margin_a']:+.2f}"
    )

    print()
    print("Agent A as player 0:")

    side = summary[
        "agent_a_as_player_0"
    ]

    print(
        f"  games={side['games']}, "
        f"W/L/T="
        f"{side['wins']}/"
        f"{side['losses']}/"
        f"{side['ties']}, "
        f"reward={side['average_reward']:.2f}, "
        f"margin={side['average_margin']:+.2f}"
    )

    print("Agent A as player 1:")

    side = summary[
        "agent_a_as_player_1"
    ]

    print(
        f"  games={side['games']}, "
        f"W/L/T="
        f"{side['wins']}/"
        f"{side['losses']}/"
        f"{side['ties']}, "
        f"reward={side['average_reward']:.2f}, "
        f"margin={side['average_margin']:+.2f}"
    )

    print()
    print(
        f"Average game time: "
        f"{summary['average_game_seconds']:.2f} s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate two Kaggriculture agents."
        )
    )

    parser.add_argument(
        "--agent-a",
        default=DEFAULT_AGENT_A,
    )

    parser.add_argument(
        "--agent-b",
        default=DEFAULT_AGENT_B,
    )

    parser.add_argument(
        "--games",
        type=int,
        default=DEFAULT_GAMES,
    )

    parser.add_argument(
        "--episode-steps",
        type=int,
        default=DEFAULT_EPISODE_STEPS,
    )

    parser.add_argument(
        "--output",
        default="artifacts/reports/evaluation_results.json",
    )

    args = parser.parse_args()

    if args.games < 1:
        raise ValueError(
            "--games must be at least 1"
        )

    print("Kaggriculture agent evaluation")
    print(f"Agent A: {args.agent_a}")
    print(f"Agent B: {args.agent_b}")
    print(f"Games:   {args.games}")
    print()

    results: list[GameResult] = []

    total_started = time.perf_counter()

    for game_number in range(
        1,
        args.games + 1,
    ):
        # Alternate seats every game.
        agent_a_player = (
            (game_number - 1) % 2
        )

        print(
            f"[{game_number:>3}/{args.games}] "
            f"A=P{agent_a_player} ... ",
            end="",
            flush=True,
        )

        try:
            result = play_game(
                args.agent_a,
                args.agent_b,
                game_number,
                agent_a_player,
                args.episode_steps,
            )

        except Exception as exc:
            print("ERROR")
            print(
                f"  {type(exc).__name__}: "
                f"{exc}"
            )
            raise

        results.append(result)

        if result.winner == "A":
            outcome = "A WIN"
        elif result.winner == "B":
            outcome = "B WIN"
        else:
            outcome = "TIE"

        print(
            f"{outcome:<5} | "
            f"A={result.reward_a:.0f} "
            f"B={result.reward_b:.0f} | "
            f"margin={result.margin_a:+.0f} | "
            f"{result.elapsed_seconds:.1f}s",
            flush=True,
        )

    summary = summarize(results)

    total_elapsed = (
        time.perf_counter()
        - total_started
    )

    report = {
        "agent_a": args.agent_a,
        "agent_b": args.agent_b,
        "episode_steps": args.episode_steps,
        "summary": summary,
        "total_elapsed_seconds": (
            total_elapsed
        ),
        "games": [
            asdict(result)
            for result in results
        ],
    }

    output_path = Path(args.output)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.write("\n")

    print_summary(
        args.agent_a,
        args.agent_b,
        summary,
    )

    print()
    print(
        f"Saved: {output_path}"
    )

    print(
        f"Total evaluation time: "
        f"{total_elapsed:.1f} s"
    )


if __name__ == "__main__":
    main()
