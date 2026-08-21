"""Run the resumable Boatlee-opponent PPO pipeline in Google Colab."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from package_rl_submission import build_rl_submission
from run_colab_pipeline import binary_sha256, submit_once
from scripts.fetch_kaggle_opponent import DEFAULT_KERNEL, fetch_opponent


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def smoke_test_submission(submission: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="kaggriculture-rl-smoke-") as directory:
        root = Path(directory)
        with tarfile.open(submission, "r:gz") as archive:
            archive.extractall(root, filter="data")
        code = (
            "from kaggle_environments import make; "
            "env=make('kaggriculture', configuration={'episodeSteps': 24}, debug=True); "
            "env.run(['main.py', 'artifacts/rl/expert_agent.py']); "
            "final=env.steps[-1]; "
            "assert all(str(s.status) == 'DONE' for s in final), final; "
            "print([(str(s.status), s.reward) for s in final])"
        )
        subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
    print("[submission] Isolated game smoke test passed", flush=True)


def copy_rl_submission(source: Path, destination: Path) -> None:
    partial = destination.with_name(f".{destination.name}.partial")
    try:
        shutil.copyfile(source, partial)
        if binary_sha256(source) != binary_sha256(partial):
            raise RuntimeError("Submission changed while copying it to Drive")
        with tarfile.open(partial, "r:gz") as archive:
            names = set(archive.getnames())
        required = {"main.py", "artifacts/rl/meta_policy.npz", "artifacts/rl/expert_agent.py"}
        if not required.issubset(names):
            raise RuntimeError("Drive copy is missing required RL files")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    print(f"[submission] Verified Drive copy: {destination}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--kernel", default=DEFAULT_KERNEL)
    parser.add_argument("--steps-per-round", type=int, default=100_000)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--target-win-rate", type=float, default=0.8)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--train-envs", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--submit-below-target", action="store_true")
    parser.add_argument("--force-submit", action="store_true")
    parser.add_argument("--competition", default="kaggriculture")
    parser.add_argument("--message", default="Boatlee expert-gated PPO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work = Path("/content/kaggriculture_rl_work")
    work.mkdir(parents=True, exist_ok=True)
    expert = work / "boatlee_v16.py"
    fetch_opponent(args.kernel, expert)

    results = args.drive_root / "rl_boatlee_v16"
    results.mkdir(parents=True, exist_ok=True)
    train_command = [
        sys.executable,
        "-u",
        "-m",
        "src.kaggriculture.rl.train_meta_controller",
        "--expert",
        str(expert),
        "--output-dir",
        str(results),
        "--steps-per-round",
        str(args.steps_per_round),
        "--max-rounds",
        str(args.max_rounds),
        "--target-win-rate",
        str(args.target_win_rate),
        "--eval-games",
        str(args.eval_games),
        "--train-envs",
        str(args.train_envs),
        "--device",
        args.device,
    ]
    run(train_command)

    report_path = results / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    local_submission = work / "submission.tar.gz"
    build_rl_submission(
        local_submission,
        results / "best_meta_policy.npz",
        expert,
    )
    smoke_test_submission(local_submission)
    drive_submission = results / "submission.tar.gz"
    copy_rl_submission(local_submission, drive_submission)
    shutil.copy2(expert, results / "boatlee_v16_source.py")

    allowed = report["target_met"] or args.submit_below_target
    if args.submit and allowed:
        submit_once(
            drive_submission,
            results / "submission_receipt.json",
            args.competition,
            args.message,
            args.force_submit,
        )
    elif args.submit:
        print(
            "[submit] Skipped: evaluation target was not reached. "
            "The verified archive remains on Drive.",
            flush=True,
        )
    print(
        json.dumps(
            {
                "submission": str(drive_submission),
                "best_metrics": report["best"],
                "target_met": report["target_met"],
                "submitted": bool(args.submit and allowed),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
