"""Search deterministic market rules through paired league simulations."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .league_policy import (
    ACTION_DIMS,
    LEAGUE_POLICY_VERSION,
    RESIDUAL_PRODUCTS,
    RULE_POLICY_KIND,
    MarketHistoryFeatures,
    NumpyLeaguePolicy,
)
from .train_league_controller import (
    _load_pool,
    evaluate_policy,
    promotion_gate,
    save_fallback_policy,
)


def save_rule_policy(path: str | Path, configuration: dict[str, Sequence[float]]) -> None:
    """Save a compact deterministic rule policy in the submission model format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "policy_kind": np.asarray(RULE_POLICY_KIND),
        "feature_count": np.asarray(MarketHistoryFeatures.feature_count),
        "action_dims": np.asarray(ACTION_DIMS),
        "policy_version": np.asarray(LEAGUE_POLICY_VERSION),
    }
    for key in (
        "minimum_price_ratios",
        "minimum_stocks",
        "sale_choices",
        "late_days",
        "demand_bonuses",
        "rising_price_bonuses",
    ):
        values = np.asarray(configuration[key])
        if values.shape != (len(RESIDUAL_PRODUCTS),):
            raise ValueError(f"{key} must contain one value per residual product")
        payload[key] = values
    np.savez_compressed(path, **payload)
    NumpyLeaguePolicy(path)


def generate_candidates(count: int, seed: int) -> list[dict[str, list[float]]]:
    """Create conservative, reproducible product-specific search candidates."""

    if count <= 0:
        raise ValueError("Candidate count must be positive")
    templates = (
        (0.85, 2, 1, 27, 0.00, 0.00),
        (1.00, 4, 2, 27, 0.00, 0.10),
        (1.15, 6, 2, 26, 0.03, 0.15),
        (1.30, 8, 3, 26, 0.05, 0.20),
        (1.50, 10, 4, 25, 0.05, 0.25),
        (1.75, 12, 4, 27, 0.08, 0.30),
    )
    candidates: list[dict[str, list[float]]] = []
    for template in templates[:count]:
        ratio, stock, choice, late_day, demand, rising = template
        candidates.append(
            {
                "minimum_price_ratios": [ratio] * 4,
                "minimum_stocks": [stock] * 4,
                "sale_choices": [choice] * 4,
                "late_days": [late_day] * 4,
                "demand_bonuses": [demand] * 4,
                "rising_price_bonuses": [rising] * 4,
            }
        )
    rng = np.random.default_rng(seed)
    while len(candidates) < count:
        candidates.append(
            {
                "minimum_price_ratios": rng.choice(
                    [0.85, 1.0, 1.15, 1.3, 1.5, 1.75], size=4
                ).tolist(),
                "minimum_stocks": rng.choice([2, 4, 6, 10, 16], size=4).tolist(),
                "sale_choices": rng.choice([1, 2, 3, 4], size=4).tolist(),
                "late_days": rng.choice([25, 26, 27, 31], size=4).tolist(),
                "demand_bonuses": rng.choice([0.0, 0.03, 0.06], size=4).tolist(),
                "rising_price_bonuses": rng.choice(
                    [0.0, 0.1, 0.2, 0.3], size=4
                ).tolist(),
            }
        )
    return candidates


def _ranking(metrics: dict[str, Any]) -> tuple[float, float]:
    return float(metrics["score_rate"]), float(metrics["mean_money_margin"])


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def search(args: argparse.Namespace) -> dict[str, Any]:
    if args.candidates <= 0 or args.finalists <= 0:
        raise ValueError("Candidate and finalist counts must be positive")
    if args.screen_opponents <= 0 or args.screen_seed_count <= 0:
        raise ValueError("Screening opponents and seeds must be positive")
    if args.final_seed_count <= 0:
        raise ValueError("Final seed count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = args.output_dir / "candidate_policies"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "counterfactual_search_report.json"
    fallback = args.output_dir / "incumbent_fallback_policy.npz"
    best_policy = args.output_dir / "best_league_policy.npz"
    save_fallback_policy(fallback)
    if not best_policy.is_file():
        shutil.copy2(fallback, best_policy)

    signature = {
        "incumbent": str(args.incumbent),
        "opponent_pool": str(args.opponent_pool),
        "candidates": args.candidates,
        "finalists": args.finalists,
        "screen_opponents": args.screen_opponents,
        "screen_seed_count": args.screen_seed_count,
        "final_seed_count": args.final_seed_count,
        "screen_seed_offset": args.screen_seed_offset,
        "final_seed_offset": args.final_seed_offset,
        "episode_steps": args.episode_steps,
        "search_seed": args.search_seed,
    }
    previous = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else {}
    )
    if previous.get("signature") != signature:
        previous = {}
        shutil.copy2(fallback, best_policy)

    opponents, veto_names = _load_pool(args.opponent_pool)
    for row in opponents:
        row["veto"] = row["slug"] in veto_names or bool(row.get("veto", False))
    final_seeds = tuple(
        range(args.final_seed_offset, args.final_seed_offset + args.final_seed_count)
    )
    print(
        f"[1/4 baseline] {len(opponents)} opponents, seeds={list(final_seeds)}, both seats",
        flush=True,
    )
    baseline = previous.get("baseline")
    if baseline:
        print("[baseline] reused from the Drive report", flush=True)
    else:
        baseline = evaluate_policy(
            lambda _: np.zeros(len(ACTION_DIMS), dtype=np.int64),
            args.incumbent,
            opponents,
            seeds=final_seeds,
            episode_steps=args.episode_steps,
        )
    by_slug = {row["slug"]: row for row in baseline["opponents"]}
    weakest = sorted(
        opponents,
        key=lambda row: (
            by_slug[row["slug"]]["score_rate"],
            by_slug[row["slug"]]["mean_money_margin"],
        ),
    )[: min(args.screen_opponents, len(opponents))]
    screen_seeds = tuple(
        range(args.screen_seed_offset, args.screen_seed_offset + args.screen_seed_count)
    )
    print(
        f"[screen set] opponents={[row['slug'] for row in weakest]}, "
        f"seeds={list(screen_seeds)}, both seats",
        flush=True,
    )
    screen_baseline = previous.get("screen_baseline")
    if screen_baseline:
        print("[screen baseline] reused from the Drive report", flush=True)
    else:
        screen_baseline = evaluate_policy(
            lambda _: np.zeros(len(ACTION_DIMS), dtype=np.int64),
            args.incumbent,
            weakest,
            seeds=screen_seeds,
            episode_steps=args.episode_steps,
        )

    configurations = generate_candidates(args.candidates, args.search_seed)
    report: dict[str, Any] = previous or {
        "schema_version": 1,
        "method": RULE_POLICY_KIND,
        "signature": signature,
        "incumbent": str(args.incumbent),
        "opponent_pool": str(args.opponent_pool),
        "baseline": baseline,
        "screen_baseline": screen_baseline,
        "screen_opponents": [row["slug"] for row in weakest],
        "screen_seeds": list(screen_seeds),
        "final_seeds": list(final_seeds),
        "screen_results": [],
        "final_results": [],
        "promoted": False,
        "best_candidate": None,
        "best_policy": str(best_policy),
    }
    print(f"[2/4 screening] {len(configurations)} candidates", flush=True)
    prior_screened = {
        row["candidate_id"]: row for row in report.get("screen_results", [])
    }
    screened: list[dict[str, Any]] = []
    for index, configuration in enumerate(configurations, 1):
        candidate_id = f"candidate_{index:03d}"
        path = candidate_dir / f"{candidate_id}.npz"
        save_rule_policy(path, configuration)
        policy = NumpyLeaguePolicy(path)
        cached = prior_screened.get(candidate_id)
        if cached and cached.get("configuration") == configuration:
            print(
                f"[screen {index}/{len(configurations)}] {candidate_id}: reused",
                flush=True,
            )
            screened.append(cached)
            continue
        print(f"[screen {index}/{len(configurations)}] {candidate_id}", flush=True)
        metrics = evaluate_policy(
            policy.predict,
            args.incumbent,
            weakest,
            seeds=screen_seeds,
            episode_steps=args.episode_steps,
        )
        row = {
            "candidate_id": candidate_id,
            "policy": str(path),
            "configuration": configuration,
            "metrics": metrics,
            "ranking": list(_ranking(metrics)),
        }
        screened.append(row)
        report["screen_results"] = screened
        _write_report(report_path, report)
        print(
            f"[screen result] {candidate_id}: score={metrics['score_rate']:.3f}, "
            f"margin={metrics['mean_money_margin']:+.1f}, "
            f"changed={metrics['effective_decisions']:,}",
            flush=True,
        )

    eligible = [
        row
        for row in screened
        if row["metrics"]["errors"] == 0
        and row["metrics"]["effective_decisions"] > 0
    ]
    eligible.sort(key=lambda row: _ranking(row["metrics"]), reverse=True)
    finalists = eligible[: min(args.finalists, len(eligible))]
    print(
        f"[3/4 held-out league] finalists={[row['candidate_id'] for row in finalists]}",
        flush=True,
    )
    report["screen_results"] = screened
    prior_final = {
        row["candidate_id"]: row for row in report.get("final_results", [])
    }
    report["final_results"] = []
    best_ranking = (
        _ranking(report["best_candidate"]["metrics"])
        if report.get("best_candidate")
        else _ranking(baseline)
    )
    for index, finalist in enumerate(finalists, 1):
        cached = prior_final.get(finalist["candidate_id"])
        if cached and cached.get("configuration") == finalist["configuration"]:
            print(
                f"[final {index}/{len(finalists)}] {finalist['candidate_id']}: reused",
                flush=True,
            )
            report["final_results"].append(cached)
            continue
        print(
            f"[final {index}/{len(finalists)}] {finalist['candidate_id']}", flush=True
        )
        policy = NumpyLeaguePolicy(finalist["policy"])
        metrics = evaluate_policy(
            policy.predict,
            args.incumbent,
            opponents,
            seeds=final_seeds,
            episode_steps=args.episode_steps,
        )
        gate = promotion_gate(baseline, metrics)
        row = {**finalist, "metrics": metrics, "promotion_gate": gate}
        report["final_results"].append(row)
        ranking = _ranking(metrics)
        print(
            f"[final result] {finalist['candidate_id']}: "
            f"score={metrics['score_rate']:.3f}, margin={metrics['mean_money_margin']:+.1f}, "
            f"passed={gate['passed']}",
            flush=True,
        )
        if gate["passed"] and ranking > best_ranking:
            shutil.copy2(finalist["policy"], best_policy)
            report["promoted"] = True
            report["best_candidate"] = row
            best_ranking = ranking
            print(f"[promotion] saved {finalist['candidate_id']}", flush=True)
        _write_report(report_path, report)

    print(
        "[4/4 result] "
        + ("counterfactual rule promoted" if report["promoted"] else "incumbent retained"),
        flush=True,
    )
    _write_report(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--opponent-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--finalists", type=int, default=2)
    parser.add_argument("--screen-opponents", type=int, default=2)
    parser.add_argument("--screen-seed-count", type=int, default=1)
    parser.add_argument("--final-seed-count", type=int, default=3)
    parser.add_argument("--screen-seed-offset", type=int, default=8_800_000)
    parser.add_argument("--final-seed-offset", type=int, default=9_200_000)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--search-seed", type=int, default=20260825)
    return parser.parse_args()


def main() -> None:
    search(parse_args())


if __name__ == "__main__":
    main()
