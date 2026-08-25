"""Run counterfactual market search through a tested Kaggle submission."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from package_league_submission import build_league_submission
from src.kaggriculture.league.evaluator import load_agent_file, materialize_main


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _required(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Incumbent archive is missing {relative}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--finalists", type=int, default=2)
    parser.add_argument("--screen-opponents", type=int, default=2)
    parser.add_argument("--screen-seed-count", type=int, default=1)
    parser.add_argument("--final-seed-count", type=int, default=3)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--message", default="Stage 3 counterfactual market search v5")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--submit-unpromoted", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool = args.drive_root / "league" / "opponent_pool.json"
    if not pool.is_file():
        raise FileNotFoundError(f"Missing opponent pool: {pool}")
    if not args.incumbent.is_file():
        raise FileNotFoundError(f"Missing incumbent archive: {args.incumbent}")
    output_dir = args.drive_root / "results" / "stage3_counterfactual_market_v5"
    work_dir = args.drive_root / "league" / "stage3_incumbent_v5"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Validating and extracting the immutable incumbent", flush=True)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    incumbent_main = materialize_main(args.incumbent, work_dir)
    load_agent_file(incumbent_main)
    incumbent_model = _required(work_dir, "artifacts/rl/meta_policy.npz")
    expert = _required(work_dir, "artifacts/rl/expert_agent.py")

    print("[2/5] Searching paired counterfactual market policies", flush=True)
    _run(
        [
            sys.executable,
            "-u",
            "-m",
            "src.kaggriculture.rl.search_counterfactual_market",
            "--incumbent",
            str(incumbent_main),
            "--opponent-pool",
            str(pool),
            "--output-dir",
            str(output_dir),
            "--candidates",
            str(args.candidates),
            "--finalists",
            str(args.finalists),
            "--screen-opponents",
            str(args.screen_opponents),
            "--screen-seed-count",
            str(args.screen_seed_count),
            "--final-seed-count",
            str(args.final_seed_count),
            "--episode-steps",
            str(args.episode_steps),
        ]
    )

    print("[3/5] Reading the held-out promotion decision", flush=True)
    report_path = output_dir / "counterfactual_search_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    promoted = bool(report.get("promoted"))
    print(
        "[gate] " + ("counterfactual rule promoted" if promoted else "incumbent retained"),
        flush=True,
    )

    print("[4/5] Building and smoke-testing submission.tar.gz", flush=True)
    archive = output_dir / "submission.tar.gz"
    selected = output_dir / "best_league_policy.npz"
    receipt = build_league_submission(archive, selected, incumbent_model, expert)
    smoke_dir = output_dir / "submission_smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_main = materialize_main(archive, smoke_dir)
    load_agent_file(smoke_main)
    receipt.update({"promoted": promoted, "incumbent_retained": not promoted})
    (output_dir / "submission_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[archive] {archive} ({receipt['bytes']:,} bytes): OK", flush=True)

    print("[5/5] Kaggle submission decision", flush=True)
    should_submit = args.submit and (promoted or args.submit_unpromoted)
    if should_submit:
        _run(
            [
                "kaggle",
                "competitions",
                "submit",
                "kaggriculture",
                "-f",
                str(archive),
                "-m",
                args.message,
            ]
        )
    else:
        reason = "submission disabled" if not args.submit else "no promoted policy"
        print(f"[submit] skipped: {reason}", flush=True)


if __name__ == "__main__":
    main()
