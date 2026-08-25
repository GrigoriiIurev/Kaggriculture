"""Turn Replay Warehouse rows into actionable loss diagnostics."""

from __future__ import annotations

import csv
import gzip
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any


LOSS_DAY_FIELDS = (
    "episode_id",
    "opponent",
    "day",
    "money_margin",
    "margin_delta",
    "our_money_delta",
    "opponent_money_delta",
    "animal_gap",
    "plant_gap",
    "hand_gap",
    "land_gap",
    "our_animals",
    "opponent_animals",
    "our_plants",
    "opponent_plants",
    "our_hands",
    "opponent_hands",
    "our_land",
    "opponent_land",
    "our_weeds",
    "our_endangered_plants",
    "our_endangered_animals",
    "shed_occupancy",
    "carried_items",
    "pass_share",
    "market_orders",
)


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _number(mapping: Mapping[str, Any], key: str) -> float:
    return float(mapping.get(key, 0) or 0)


def _inventory_total(mapping: Mapping[str, Any]) -> int:
    return sum(max(0, int(value or 0)) for value in mapping.values())


def _pass_share(actions: Mapping[str, Any]) -> float:
    total = sum(max(0, int(value or 0)) for value in actions.values())
    return max(0, int(actions.get("PASS", 0) or 0)) / total if total else 0.0


def _daily_row(row: Mapping[str, Any], previous_margin: float | None) -> dict[str, Any]:
    end = row["end"]
    ours = end["me"]
    opponent = end["opponent"]
    peak = row.get("peak", {}) or {}
    our_scale = peak.get("me", ours)
    opponent_scale = peak.get("opponent", opponent)
    private = end.get("private", {}) or {}
    margin = _number(ours, "money") - _number(opponent, "money")
    market_operations = row.get("market_operations", {}) or {}
    result = {
        "episode_id": int(row["episode_id"]),
        "opponent": str(row["opponent"]),
        "day": int(row["day"]),
        "money_margin": margin,
        "margin_delta": margin - previous_margin if previous_margin is not None else margin,
        "our_money_delta": float(row.get("money_delta", 0) or 0),
        "opponent_money_delta": _number(opponent, "money")
        - _number(row["start"]["opponent"], "money"),
        "animal_gap": _number(opponent_scale, "animal_total")
        - _number(our_scale, "animal_total"),
        "plant_gap": _number(opponent_scale, "plant_total")
        - _number(our_scale, "plant_total"),
        "hand_gap": _number(opponent_scale, "hands") - _number(our_scale, "hands"),
        "land_gap": _number(opponent_scale, "land") - _number(our_scale, "land"),
        "our_animals": _number(our_scale, "animal_total"),
        "opponent_animals": _number(opponent_scale, "animal_total"),
        "our_plants": _number(our_scale, "plant_total"),
        "opponent_plants": _number(opponent_scale, "plant_total"),
        "our_hands": _number(our_scale, "hands"),
        "opponent_hands": _number(opponent_scale, "hands"),
        "our_land": _number(our_scale, "land"),
        "opponent_land": _number(opponent_scale, "land"),
        "our_weeds": _number(ours, "weeds"),
        "our_endangered_plants": _number(ours, "endangered_plants"),
        "our_endangered_animals": _number(ours, "endangered_animals"),
        "shed_occupancy": _inventory_total(private.get("shed", {}) or {}),
        "carried_items": _inventory_total(private.get("carried", {}) or {}),
        "pass_share": _pass_share(row.get("field_actions", {}) or {}),
        "market_orders": sum(
            max(0, int(value or 0)) for value in market_operations.values()
        ),
    }
    return result


def _first_deficit(days: list[dict[str, Any]], threshold: float = -1_000) -> int:
    for index, row in enumerate(days):
        if float(row["money_margin"]) > threshold:
            continue
        remaining = days[index:]
        if sum(float(item["money_margin"]) <= threshold for item in remaining) >= max(
            1, len(remaining) // 2
        ):
            return int(row["day"])
    return int(days[-1]["day"])


def _diagnose(
    days: list[dict[str, Any]], win_benchmark: Mapping[str, float]
) -> tuple[str, dict[str, float], list[str]]:
    first_day = _first_deficit(days)
    relevant = [row for row in days if int(row["day"]) <= first_day] or days[:1]
    pivot = relevant[-1]
    scores: Counter[str] = Counter()
    evidence: list[str] = []
    if float(pivot["animal_gap"]) >= 3:
        scores["production_scale"] += 4
        evidence.append(f"animal gap {float(pivot['animal_gap']):+.0f} by day {first_day}")
    if float(pivot["plant_gap"]) >= 4:
        scores["crop_scale"] += 3
        evidence.append(f"plant gap {float(pivot['plant_gap']):+.0f} by day {first_day}")
    if float(pivot["hand_gap"]) >= 2:
        scores["workforce_scale"] += 3
        evidence.append(f"hand gap {float(pivot['hand_gap']):+.0f} by day {first_day}")
    if float(pivot["land_gap"]) >= 1:
        scores["expansion_timing"] += 2
        evidence.append(f"land gap {float(pivot['land_gap']):+.0f} by day {first_day}")
    maintenance_days = sum(
        float(row["our_endangered_plants"]) + float(row["our_endangered_animals"]) > 0
        for row in days
    )
    maintenance_threshold = max(
        2.0, float(win_benchmark.get("maintenance_risk_days", 0)) + 2.0
    )
    if maintenance_days >= maintenance_threshold:
        scores["maintenance_failures"] += 4
        evidence.append(f"endangered crops/animals on {maintenance_days} days")
    storage_days = sum(float(row["shed_occupancy"]) >= 90 for row in days)
    storage_threshold = max(
        2.0, float(win_benchmark.get("shed_pressure_days", 0)) + 1.0
    )
    if storage_days >= storage_threshold:
        scores["storage_pressure"] += 3
        evidence.append(f"shed at least 90% full on {storage_days} days")
    carried_days = sum(float(row["carried_items"]) >= 20 for row in days)
    if carried_days >= 2:
        scores["transport_backlog"] += 2
        evidence.append(f"at least 20 carried items on {carried_days} days")
    mean_pass = statistics.fmean(float(row["pass_share"]) for row in days)
    if mean_pass >= 0.45:
        scores["worker_utilization"] += 2
        evidence.append(f"mean PASS share {mean_pass:.0%}")
    if first_day >= 23:
        scores["late_realization"] += 3
        evidence.append(f"lasting deficit began only on day {first_day}")
    if not scores:
        scores["execution_or_opponent_timing"] = 1
        evidence.append("no dominant public scale or maintenance gap detected")
    priority = {
        "maintenance_failures": 0,
        "production_scale": 1,
        "workforce_scale": 2,
        "crop_scale": 3,
        "expansion_timing": 4,
        "storage_pressure": 5,
        "transport_backlog": 6,
        "worker_utilization": 7,
        "late_realization": 8,
        "execution_or_opponent_timing": 9,
    }
    primary = min(scores, key=lambda name: (-scores[name], priority[name]))
    return primary, dict(scores), evidence


def _episode_diagnostic(
    days: list[dict[str, Any]], win_benchmark: Mapping[str, float]
) -> dict[str, Any]:
    primary, scores, evidence = _diagnose(days, win_benchmark)
    largest_swing = min(days, key=lambda row: float(row["margin_delta"]))
    final = days[-1]
    deficit_day = _first_deficit(days)
    deficit = next(row for row in days if int(row["day"]) >= deficit_day)
    return {
        "episode_id": int(final["episode_id"]),
        "opponent": str(final["opponent"]),
        "final_margin": float(final["money_margin"]),
        "first_lasting_deficit_day": deficit_day,
        "largest_negative_swing_day": int(largest_swing["day"]),
        "largest_negative_swing": float(largest_swing["margin_delta"]),
        "primary_cause": primary,
        "cause_scores": scores,
        "evidence": evidence,
        "peak_animal_gap": max(float(row["animal_gap"]) for row in days),
        "peak_plant_gap": max(float(row["plant_gap"]) for row in days),
        "peak_hand_gap": max(float(row["hand_gap"]) for row in days),
        "peak_land_gap": max(float(row["land_gap"]) for row in days),
        "shed_pressure_days": sum(float(row["shed_occupancy"]) >= 90 for row in days),
        "maintenance_risk_days": sum(
            float(row["our_endangered_plants"]) + float(row["our_endangered_animals"]) > 0
            for row in days
        ),
        "mean_pass_share": statistics.fmean(float(row["pass_share"]) for row in days),
        "animal_gap_at_deficit": float(deficit["animal_gap"]),
        "plant_gap_at_deficit": float(deficit["plant_gap"]),
        "hand_gap_at_deficit": float(deficit["hand_gap"]),
        "land_gap_at_deficit": float(deficit["land_gap"]),
    }


def _recommendations(causes: Mapping[str, int]) -> list[str]:
    messages = {
        "production_scale": "Test earlier animal purchases and feeding capacity.",
        "crop_scale": "Test earlier or denser crop planting without reducing maintenance.",
        "workforce_scale": "Test earlier hiring when the task backlog exceeds active workers.",
        "expansion_timing": "Test earlier land purchases only when workers can use the area.",
        "maintenance_failures": "Prioritize watering and feeding before development tasks.",
        "storage_pressure": "Return carried goods and liquidate before the shed reaches 90%.",
        "transport_backlog": "Allocate a dedicated shed-return worker during harvest peaks.",
        "worker_utilization": "Inspect PASS-heavy days for route stalls or missing tasks.",
        "late_realization": "Compare endgame liquidation and last-investment cutoffs.",
        "execution_or_opponent_timing": "Inspect the largest negative swing turn-by-turn.",
    }
    ranked = sorted(causes, key=lambda name: (-int(causes[name]), name))
    return [messages[name] for name in ranked[:4]]


def _cohort_comparison(
    grouped: Mapping[int, list[dict[str, Any]]], results: Mapping[int, str]
) -> dict[str, dict[str, float]]:
    cohorts: dict[str, list[dict[str, float]]] = defaultdict(list)
    for episode_id, episode_rows in grouped.items():
        previous_margin: float | None = None
        days = []
        for row in sorted(episode_rows, key=lambda item: int(item["day"])):
            normalized = _daily_row(row, previous_margin)
            previous_margin = float(normalized["money_margin"])
            days.append(normalized)
        if not days:
            continue
        cohorts[results[episode_id]].append(
            {
                "peak_animals": max(float(row["our_animals"]) for row in days),
                "peak_plants": max(float(row["our_plants"]) for row in days),
                "peak_hands": max(float(row["our_hands"]) for row in days),
                "peak_land": max(float(row["our_land"]) for row in days),
                "mean_pass_share": statistics.fmean(
                    float(row["pass_share"]) for row in days
                ),
                "shed_pressure_days": float(
                    sum(float(row["shed_occupancy"]) >= 90 for row in days)
                ),
                "maintenance_risk_days": float(
                    sum(
                        float(row["our_endangered_plants"])
                        + float(row["our_endangered_animals"])
                        > 0
                        for row in days
                    )
                ),
            }
        )
    return {
        result: {
            "games": float(len(items)),
            **{
                field: statistics.fmean(item[field] for item in items)
                for field in (
                    "peak_animals",
                    "peak_plants",
                    "peak_hands",
                    "peak_land",
                    "mean_pass_share",
                    "shed_pressure_days",
                    "maintenance_risk_days",
                )
            },
        }
        for result, items in sorted(cohorts.items())
        if items
    }


def _markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    deficit_label = (
        str(summary["median_first_deficit_day"])
        if summary["median_first_deficit_day"] is not None
        else "n/a"
    )
    lines = [
        "# Kaggriculture Loss Replay Analysis",
        "",
        f"- Team: `{report['team_name']}`",
        f"- Games analyzed: **{summary['games']}**",
        f"- Losses diagnosed: **{summary['losses']}**",
        f"- Mean loss margin: **{summary['mean_loss_margin']:+.1f}**",
        f"- Median first lasting deficit: **day {deficit_label}**",
        f"- Median gaps at that point: **{summary['median_gaps_at_deficit']['animals']:+.1f} animals, "
        f"{summary['median_gaps_at_deficit']['plants']:+.1f} plants, "
        f"{summary['median_gaps_at_deficit']['hands']:+.1f} hands, "
        f"{summary['median_gaps_at_deficit']['land']:+.1f} land**",
        "",
        "## Primary Loss Causes",
        "",
        "| Diagnostic | Losses |",
        "|---|---:|",
    ]
    for name, count in sorted(
        summary["primary_causes"].items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"| `{name}` | {count} |")
    lines.extend(
        [
            "",
            "## Wins Versus Losses",
            "",
            "| Result | Games | Animals | Plants | Hands | Land | PASS | Shed-risk days | Maintenance-risk days |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result, row in summary["cohort_comparison"].items():
        lines.append(
            f"| {result} | {row['games']:.0f} | {row['peak_animals']:.1f} "
            f"| {row['peak_plants']:.1f} | {row['peak_hands']:.1f} "
            f"| {row['peak_land']:.1f} | {row['mean_pass_share']:.1%} "
            f"| {row['shed_pressure_days']:.1f} | {row['maintenance_risk_days']:.1f} |"
        )
    lines.extend(["", "## Recommended Experiments", ""])
    lines.extend(f"- {message}" for message in summary["recommended_experiments"])
    lines.extend(
        [
            "",
            "## Worst Losses",
            "",
            "| Episode | Opponent | Margin | First deficit | Largest swing | Diagnostic |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in summary["worst_losses"]:
        lines.append(
            f"| {row['episode_id']} | {row['opponent']} | {row['final_margin']:+.0f} "
            f"| {row['first_lasting_deficit_day']} | day {row['largest_negative_swing_day']} "
            f"({row['largest_negative_swing']:+.0f}) | `{row['primary_cause']}` |"
        )
    lines.extend(
        [
            "",
            "Diagnostics are evidence-ranking heuristics, not proof of causality. "
            "Use the referenced episode and day for turn-level inspection.",
            "",
        ]
    )
    return "\n".join(lines)


def build_loss_replay_analysis(
    daily_macro_path: str | Path,
    output_directory: str | Path,
    team_name: str,
) -> dict[str, Any]:
    """Build per-day and per-loss diagnostics from Replay Warehouse data."""

    daily_macro_path = Path(daily_macro_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in _read_jsonl_gz(daily_macro_path)
        if str(row.get("team")) == team_name and row.get("episode_type") == "public"
    ]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    results: dict[int, str] = {}
    for row in rows:
        episode_id = int(row["episode_id"])
        grouped[episode_id].append(row)
        results[episode_id] = str(row.get("result", ""))

    cohort_comparison = _cohort_comparison(grouped, results)
    win_benchmark = cohort_comparison.get("win", {})
    loss_days: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for episode_id, episode_rows in sorted(grouped.items()):
        if results[episode_id] != "loss":
            continue
        previous_margin: float | None = None
        normalized_days = []
        for row in sorted(episode_rows, key=lambda item: int(item["day"])):
            normalized = _daily_row(row, previous_margin)
            previous_margin = float(normalized["money_margin"])
            normalized_days.append(normalized)
            loss_days.append(normalized)
        if normalized_days:
            diagnostics.append(_episode_diagnostic(normalized_days, win_benchmark))

    causes = Counter(row["primary_cause"] for row in diagnostics)
    margins = [float(row["final_margin"]) for row in diagnostics]
    summary = {
        "games": len(grouped),
        "losses": len(diagnostics),
        "mean_loss_margin": statistics.fmean(margins) if margins else 0.0,
        "median_first_deficit_day": statistics.median(
            [int(row["first_lasting_deficit_day"]) for row in diagnostics]
        )
        if diagnostics
        else None,
        "median_gaps_at_deficit": {
            label: statistics.median(
                float(row[field]) for row in diagnostics
            )
            if diagnostics
            else 0.0
            for label, field in {
                "animals": "animal_gap_at_deficit",
                "plants": "plant_gap_at_deficit",
                "hands": "hand_gap_at_deficit",
                "land": "land_gap_at_deficit",
            }.items()
        },
        "primary_causes": dict(sorted(causes.items())),
        "cohort_comparison": cohort_comparison,
        "recommended_experiments": _recommendations(causes),
        "worst_losses": sorted(
            diagnostics, key=lambda row: float(row["final_margin"])
        )[:10],
    }
    report = {
        "version": 1,
        "team_name": team_name,
        "source": str(daily_macro_path),
        "summary": summary,
        "losses": diagnostics,
        "files": {
            "loss_days": "loss_days.csv",
            "diagnostics": "loss_diagnostics.json",
            "report": "loss_report.md",
        },
    }
    with (output_directory / "loss_days.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=LOSS_DAY_FIELDS)
        writer.writeheader()
        writer.writerows(loss_days)
    (output_directory / "loss_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_directory / "loss_report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(
        f"[loss analysis] {len(grouped)} games, {len(diagnostics)} losses, "
        f"causes={dict(causes)}",
        flush=True,
    )
    return report
