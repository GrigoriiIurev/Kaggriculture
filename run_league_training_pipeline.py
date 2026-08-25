"""Run Stage 3 from a Stage 2 pool through a tested Kaggle archive."""

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


def _find_artifact(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Incumbent archive is missing {relative}")
    return path


def _write_replay_context(drive_root: Path, output_dir: Path) -> None:
    candidates = [
        drive_root / "replay_warehouse" / "latest_analysis.json",
        drive_root / "replay_warehouse" / "analysis" / "latest_analysis.json",
    ]
    source = next((path for path in candidates if path.is_file()), None)
    payload = {
        "available": source is not None,
        "source": str(source) if source else None,
        "purpose": (
            "Stage 1 replay diagnostics are retained for curriculum review; "
            "the Stage 3 policy is trained only through legal local simulations."
        ),
    }
    if source:
        payload["analysis"] = json.loads(source.read_text(encoding="utf-8"))
    (output_dir / "replay_context.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--steps-per-round", type=int, default=10_000)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--train-envs", type=int, default=2)
    parser.add_argument("--eval-seed-count", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--message", default="Stage 3 league market residual")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--submit-unpromoted", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool = args.drive_root / "league" / "opponent_pool.json"
    if not pool.is_file():
        raise FileNotFoundError(
            f"Missing {pool}. Complete the Stage 2 opponent league first."
        )
    if not args.incumbent.is_file():
        raise FileNotFoundError(f"Missing incumbent archive: {args.incumbent}")

    output_dir = args.drive_root / "results" / "stage3_league_market_v4"
    work_dir = args.drive_root / "league" / "stage3_incumbent_v4"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[1/5] Validating and extracting the immutable incumbent", flush=True)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    incumbent_main = materialize_main(args.incumbent, work_dir)
    load_agent_file(incumbent_main)
    incumbent_model = _find_artifact(work_dir, "artifacts/rl/meta_policy.npz")
    expert = _find_artifact(work_dir, "artifacts/rl/expert_agent.py")
    _write_replay_context(args.drive_root, output_dir)

    print("[2/5] Training against the weighted opponent league", flush=True)
    _run(
        [
            sys.executable,
            "-u",
            "-m",
            "src.kaggriculture.rl.train_league_controller",
            "--incumbent",
            str(incumbent_main),
            "--opponent-pool",
            str(pool),
            "--output-dir",
            str(output_dir),
            "--steps-per-round",
            str(args.steps_per_round),
            "--max-rounds",
            str(args.max_rounds),
            "--train-envs",
            str(args.train_envs),
            "--eval-seed-count",
            str(args.eval_seed_count),
            "--episode-steps",
            str(args.episode_steps),
            "--device",
            args.device,
        ]
    )

    print("[3/5] Reading the promotion decision", flush=True)
    report_path = output_dir / "league_training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    promoted = bool(report.get("promoted"))
    selected = output_dir / "best_league_policy.npz"
    print(
        "[gate] " + ("new residual promoted" if promoted else "incumbent retained"),
        flush=True,
    )

    print("[4/5] Building and smoke-testing submission.tar.gz", flush=True)
    archive = output_dir / "submission.tar.gz"
    receipt = build_league_submission(
        archive, selected, incumbent_model, expert
    )
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
        reason = "submission disabled" if not args.submit else "no promoted model"
        print(f"[submit] skipped: {reason}", flush=True)


if __name__ == "__main__":
    main()
