"""Build the public opponent pool and run a resumable local league."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from src.kaggriculture.league.evaluator import (
    AgentSpec,
    evaluate_league,
    load_agent_file,
    materialize_main,
)
from src.kaggriculture.league.notebook_source import fetch_notebook_agents


def _parse_seeds(raw: str, count: int, start: int) -> tuple[int, ...]:
    if raw:
        seeds = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
        if not seeds:
            raise ValueError("--seeds did not contain a number")
        return seeds
    return tuple(range(start, start + count))


def _write_report(summary: dict[str, object], output: Path) -> None:
    gate = summary["promotion_gate"]
    lines = [
        "# Kaggriculture League Report",
        "",
        f"Completed games: **{summary['completed_games']} / {summary['requested_games']}**",
        f"Promotion gate: **{'PASS' if gate['passed'] else 'FAIL'}**",
        f"Challenger score rate: **{gate['challenger_score_rate']:.1%}**",
        "",
        "## Ranking",
        "",
        "| Rank | Agent | Role | BT | W-L-T | Score rate | Mean margin |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["rankings"]:
        lines.append(
            f"| {row['rank']} | {row['title']} | {row['role']} | "
            f"{row['bt_rating']:.1f} | {row['wins']}-{row['losses']}-{row['ties']} | "
            f"{row['score_rate']:.1%} | {row['mean_margin']:+,.1f} |"
        )
    lines.extend(
        [
            "",
            "## Veto opponents",
            "",
            ", ".join(f"`{name}`" for name in summary["veto_opponents"]),
            "",
            "A future candidate must beat the incumbent gate and must not regress "
            "against this veto set.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--challenger", type=Path)
    parser.add_argument("--challenger-name", default="our_agent")
    parser.add_argument("--refresh-opponents", action="store_true")
    parser.add_argument("--full-round-robin", action="store_true")
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=8100)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--request-timeout", type=int, default=90)
    parser.add_argument("--kaggle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.drive_root / "league"
    agents_dir = root / "agents"
    results_dir = root / "results"
    root.mkdir(parents=True, exist_ok=True)

    print("[1/4] Fetching and extracting public notebook agents", flush=True)
    manifest = fetch_notebook_agents(
        agents_dir,
        refresh=args.refresh_opponents,
        timeout=args.request_timeout,
        kaggle_executable=args.kaggle,
    )
    specs = [
        AgentSpec(
            slug=record["slug"],
            path=Path(record["path"]),
            title=record["title"],
            role=record["role"],
        )
        for record in manifest["agents"]
    ]

    print("[2/4] Loading every extracted agent", flush=True)
    for index, spec in enumerate(specs, start=1):
        load_agent_file(spec.path)
        print(f"[load {index}/{len(specs)}] {spec.slug}: OK", flush=True)

    challenger_name = None
    if args.challenger:
        payload_digest = hashlib.sha256(args.challenger.read_bytes()).hexdigest()[:16]
        challenger_root = root / "challengers" / payload_digest
        challenger_path = materialize_main(args.challenger, challenger_root)
        load_agent_file(challenger_path)
        specs.append(
            AgentSpec(
                slug=args.challenger_name,
                path=challenger_path,
                title=args.challenger_name,
                role="challenger",
            )
        )
        challenger_name = args.challenger_name
        print(f"[challenger] {challenger_path}: OK", flush=True)
    elif not args.full_round_robin:
        raise ValueError("Pass --challenger or enable --full-round-robin")

    seeds = _parse_seeds(args.seeds, args.seed_count, args.seed_start)
    print(
        f"[3/4] Running seat-balanced league on seeds {list(seeds)}", flush=True
    )
    summary = evaluate_league(
        specs,
        results_dir,
        seeds=seeds,
        challenger=challenger_name,
        full_round_robin=args.full_round_robin,
        episode_steps=args.episode_steps,
        max_games=args.max_games,
    )

    print("[4/4] Writing league report and training-pool manifest", flush=True)
    _write_report(summary, results_dir / "report.md")
    measured_ratings = {
        row["slug"]: float(row["bt_rating"])
        for row in summary["rankings"]
        if row["slug"] != challenger_name
    }
    ratings = {
        record["slug"]: measured_ratings.get(record["slug"], 1500.0)
        for record in manifest["agents"]
    }
    highest = max(ratings.values(), default=1500.0)
    raw_weights = {
        slug: math.exp((rating - highest) / 400.0)
        for slug, rating in ratings.items()
    }
    weight_total = sum(raw_weights.values()) or 1.0
    uniform = 1.0 / max(1, len(raw_weights))
    training_opponents = []
    for record in manifest["agents"]:
        learned = raw_weights.get(record["slug"], 0.0) / weight_total
        training_opponents.append(
            {
                "slug": record["slug"],
                "path": record["path"],
                "sha256": record["sha256"],
                "sampling_weight": round(0.7 * learned + 0.3 * uniform, 6),
                "veto": record["slug"] in summary["veto_opponents"],
            }
        )
    pool = {
        "schema_version": 1,
        "opponents": manifest["agents"],
        "training_opponents": training_opponents,
        "veto_opponents": summary["veto_opponents"],
        "ranking": summary["rankings"],
        "promotion_gate": summary["promotion_gate"],
        "files": {
            "games": str(results_dir / "games.jsonl"),
            "pairs": str(results_dir / "pair_results.csv"),
            "rankings": str(results_dir / "rankings.csv"),
            "report": str(results_dir / "report.md"),
        },
    }
    (root / "opponent_pool.json").write_text(
        json.dumps(pool, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(pool, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
