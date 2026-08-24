"""Seat-balanced Kaggriculture league with resumable match storage."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class AgentSpec:
    slug: str
    path: Path
    title: str = ""
    role: str = ""


def _safe_archive_member(name: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def materialize_main(path: str | Path, directory: str | Path) -> Path:
    """Return a Python entrypoint from main.py or a Kaggle tar archive."""

    path = Path(path)
    if path.suffix == ".py":
        return path
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if any(not _safe_archive_member(member.name) for member in members):
            raise ValueError(f"Unsafe path in archive {path}")
        main = next((member for member in members if member.name == "main.py"), None)
        if main is None:
            raise ValueError(f"Archive {path} has no root main.py")
        archive.extractall(directory, filter="data")
    return directory / "main.py"


def load_agent_file(path: str | Path) -> Callable[..., dict[str, Any]]:
    """Import an agent entrypoint under an isolated module name."""

    path = Path(path)
    name = f"kaggriculture_league_{path.stem}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise AttributeError(f"{path} does not define callable agent")
    return agent


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_key(
    a: AgentSpec,
    b: AgentSpec,
    seed: int,
    seat_a: int,
    episode_steps: int,
) -> str:
    return (
        f"{a.slug}|{_file_digest(a.path)}|{b.slug}|{_file_digest(b.path)}|"
        f"{seed}|{seat_a}|{episode_steps}"
    )


def _play_game(
    a: AgentSpec,
    b: AgentSpec,
    seed: int,
    seat_a: int,
    episode_steps: int,
) -> dict[str, Any]:
    from kaggle_environments import make

    started = time.time()
    error = ""
    rewards = [0.0, 0.0]
    statuses = ["ERROR", "ERROR"]
    try:
        agent_a = load_agent_file(a.path)
        agent_b = load_agent_file(b.path)
        seats = [agent_a, agent_b] if seat_a == 0 else [agent_b, agent_a]
        env = make(
            "kaggriculture",
            configuration={"episodeSteps": episode_steps, "seed": seed},
            debug=False,
        )
        env.run(seats)
        final = env.steps[-1]
        rewards = [float(state.reward or 0) for state in final]
        statuses = [str(state.status) for state in final]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    reward_a = rewards[seat_a]
    reward_b = rewards[1 - seat_a]
    margin = reward_a - reward_b
    if error or any(status != "DONE" for status in statuses):
        outcome = "error"
    elif margin > 0:
        outcome = "win"
    elif margin < 0:
        outcome = "loss"
    else:
        outcome = "tie"
    return {
        "key": _result_key(a, b, seed, seat_a, episode_steps),
        "agent_a": a.slug,
        "agent_b": b.slug,
        "seed": seed,
        "seat_a": seat_a,
        "reward_a": reward_a,
        "reward_b": reward_b,
        "margin_a": margin,
        "outcome_a": outcome,
        "status_a": statuses[seat_a],
        "status_b": statuses[1 - seat_a],
        "error": error,
        "seconds": round(time.time() - started, 3),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        output.write("\n")


def _pair_rows(games: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for game in games:
        grouped.setdefault((game["agent_a"], game["agent_b"]), []).append(game)
    rows = []
    for (a, b), values in sorted(grouped.items()):
        valid = [value for value in values if value["outcome_a"] != "error"]
        margins = [float(value["margin_a"]) for value in valid]
        rows.append(
            {
                "agent_a": a,
                "agent_b": b,
                "games": len(values),
                "valid_games": len(valid),
                "wins_a": sum(value["outcome_a"] == "win" for value in valid),
                "wins_b": sum(value["outcome_a"] == "loss" for value in valid),
                "ties": sum(value["outcome_a"] == "tie" for value in valid),
                "errors": len(values) - len(valid),
                "mean_margin_a": statistics.fmean(margins) if margins else 0.0,
                "median_margin_a": statistics.median(margins) if margins else 0.0,
            }
        )
    return rows


def bradley_terry(
    pair_rows: Iterable[dict[str, Any]],
    *,
    iterations: int = 10_000,
    tolerance: float = 1e-10,
    prior: float = 0.5,
) -> dict[str, float]:
    """Fit regularized Bradley-Terry ratings; a tie is half a win each."""

    pairs: dict[tuple[str, str], tuple[float, float]] = {}
    for row in pair_rows:
        key = (str(row["agent_a"]), str(row["agent_b"]))
        wins_a, wins_b = pairs.get(key, (0.0, 0.0))
        pairs[key] = (
            wins_a + float(row["wins_a"]) + 0.5 * float(row["ties"]),
            wins_b + float(row["wins_b"]) + 0.5 * float(row["ties"]),
        )
    names = sorted({name for pair in pairs for name in pair})
    if not names:
        return {}
    strength = {name: 1.0 for name in names}
    wins = {name: 0.0 for name in names}
    games: dict[str, list[tuple[str, float]]] = {name: [] for name in names}
    for (a, b), (wins_a, wins_b) in pairs.items():
        wins[a] += wins_a
        wins[b] += wins_b
        total = wins_a + wins_b
        games[a].append((b, total))
        games[b].append((a, total))

    for _ in range(iterations):
        delta = 0.0
        for name in names:
            denominator = prior / (prior + 1.0) * 2.0
            denominator += sum(
                total / (strength[name] + strength[other])
                for other, total in games[name]
            )
            updated = (wins[name] + prior) / max(denominator, 1e-12)
            delta = max(delta, abs(updated - strength[name]) / max(updated, 1e-12))
            strength[name] = updated
        geometric_mean = math.exp(
            sum(math.log(max(value, 1e-12)) for value in strength.values())
            / len(strength)
        )
        strength = {name: value / geometric_mean for name, value in strength.items()}
        if delta < tolerance:
            break
    return {
        name: 1500.0 + 400.0 * math.log10(max(value, 1e-12))
        for name, value in strength.items()
    }


def _rankings(
    games: list[dict[str, Any]], specs: dict[str, AgentSpec]
) -> list[dict[str, Any]]:
    pairs = _pair_rows(games)
    ratings = bradley_terry(pairs)
    rows = []
    for name, rating in ratings.items():
        relevant = [
            game
            for game in games
            if game["outcome_a"] != "error"
            and name in {game["agent_a"], game["agent_b"]}
        ]
        wins = sum(
            (game["outcome_a"] == "win" and game["agent_a"] == name)
            or (game["outcome_a"] == "loss" and game["agent_b"] == name)
            for game in relevant
        )
        losses = sum(
            (game["outcome_a"] == "loss" and game["agent_a"] == name)
            or (game["outcome_a"] == "win" and game["agent_b"] == name)
            for game in relevant
        )
        ties = len(relevant) - wins - losses
        margins = [
            float(game["margin_a"])
            if game["agent_a"] == name
            else -float(game["margin_a"])
            for game in relevant
        ]
        spec = specs[name]
        rows.append(
            {
                "rank": 0,
                "slug": name,
                "title": spec.title or name,
                "role": spec.role,
                "bt_rating": round(rating, 1),
                "games": len(relevant),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "score_rate": round((wins + 0.5 * ties) / len(relevant), 4)
                if relevant
                else 0.0,
                "mean_margin": round(statistics.fmean(margins), 1) if margins else 0.0,
            }
        )
    rows.sort(key=lambda row: (row["bt_rating"], row["score_rate"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_league(
    agents: list[AgentSpec],
    output_directory: str | Path,
    *,
    seeds: Iterable[int],
    challenger: str | None = None,
    full_round_robin: bool = False,
    episode_steps: int = 720,
    max_games: int = 0,
) -> dict[str, Any]:
    """Evaluate requested pairings, checkpointing after every completed game."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    specs = {agent.slug: agent for agent in agents}
    if len(specs) != len(agents):
        raise ValueError("Agent slugs must be unique")
    if challenger and challenger not in specs:
        raise ValueError(f"Unknown challenger {challenger!r}")
    if not full_round_robin and not challenger:
        raise ValueError("challenger is required unless full_round_robin is enabled")

    pairings = list(combinations(agents, 2)) if full_round_robin else [
        (specs[challenger], opponent)
        for opponent in agents
        if opponent.slug != challenger
    ]
    seeds = tuple(int(seed) for seed in seeds)
    requested = [
        (a, b, seed, seat_a)
        for a, b in pairings
        for seed in seeds
        for seat_a in (0, 1)
    ]
    results_path = output_directory / "games.jsonl"
    cached = _read_jsonl(results_path)
    by_key = {record["key"]: record for record in cached}
    pending = [
        item
        for item in requested
        if _result_key(*item, episode_steps) not in by_key
    ]
    uncached_count = len(pending)
    if max_games:
        pending = pending[:max_games]
    print(
        f"[league] {len(requested)} games requested, "
        f"{len(requested) - uncached_count} available from cache, "
        f"{len(pending)} to play, {uncached_count - len(pending)} deferred",
        flush=True,
    )
    for index, (a, b, seed, seat_a) in enumerate(pending, start=1):
        print(
            f"[game {index}/{len(pending)}] {a.slug} vs {b.slug}, "
            f"seed={seed}, {a.slug} seat={seat_a}",
            flush=True,
        )
        record = _play_game(a, b, seed, seat_a, episode_steps)
        _append_jsonl(results_path, record)
        by_key[record["key"]] = record
        print(
            f"[game {index}/{len(pending)}] {record['outcome_a']} "
            f"margin={record['margin_a']:+,.0f} ({record['seconds']:.1f}s)",
            flush=True,
        )

    selected = [
        by_key[_result_key(*item, episode_steps)]
        for item in requested
        if _result_key(*item, episode_steps) in by_key
    ]
    pair_rows = _pair_rows(selected)
    rankings = _rankings(selected, specs)
    _write_csv(output_directory / "pair_results.csv", pair_rows)
    _write_csv(output_directory / "rankings.csv", rankings)

    veto = [row["slug"] for row in rankings if row["slug"] != challenger][:3]
    challenger_pairs = [
        row
        for row in pair_rows
        if challenger and challenger in {row["agent_a"], row["agent_b"]}
    ]
    valid = sum(int(row["valid_games"]) for row in challenger_pairs)
    score = sum(
        int(row["wins_a"] if row["agent_a"] == challenger else row["wins_b"])
        + 0.5 * int(row["ties"])
        for row in challenger_pairs
    )
    errors = sum(int(row["errors"]) for row in challenger_pairs)
    challenger_score_rate = score / valid if valid else 0.0
    gate = {
        "complete": len(selected) == len(requested),
        "runtime_errors": errors,
        "challenger_score_rate": round(challenger_score_rate, 4),
        "minimum_score_rate": 0.5,
        "passed": len(selected) == len(requested)
        and errors == 0
        and challenger_score_rate >= 0.5,
    }
    summary = {
        "schema_version": 1,
        "challenger": challenger,
        "full_round_robin": full_round_robin,
        "seeds": list(seeds),
        "episode_steps": episode_steps,
        "requested_games": len(requested),
        "completed_games": len(selected),
        "veto_opponents": veto,
        "promotion_gate": gate,
        "rankings": rankings,
        "pair_results": pair_rows,
    }
    (output_directory / "league_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
