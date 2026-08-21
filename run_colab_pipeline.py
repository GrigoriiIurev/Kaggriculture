"""Run the resumable Colab training-to-submission pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path


REQUIRED_DATASET_FILES = (
    "transitions.jsonl.gz",
    "manifest.json",
    "worker_dataset.jsonl.gz",
    "worker_manifest.json",
)


def run(command: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def complete_dataset(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED_DATASET_FILES)


def restore_checkpoint(archive: Path, directory: Path) -> bool:
    if directory.is_dir() and any(directory.iterdir()):
        return False
    if not archive.is_file():
        return False
    directory.parent.mkdir(parents=True, exist_ok=True)
    print(f"[checkpoint] Restoring {archive.name}...", flush=True)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for number, member in enumerate(members, start=1):
            if number == 1 or number % 10 == 0 or number == len(members):
                print(
                    f"[checkpoint restore] {number:,}/{len(members):,}: "
                    f"{member.name}",
                    flush=True,
                )
            bundle.extract(member, directory.parent, filter="data")
    print(f"[checkpoint] Restored {directory}", flush=True)
    return True


def save_checkpoint(
    directory: Path,
    archive: Path,
    temporary_directory: Path = Path("/content"),
) -> None:
    if archive.is_file():
        print(f"[checkpoint] Reusing {archive}", flush=True)
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    local_archive = (
        temporary_directory / f"kaggriculture-{uuid.uuid4().hex}.tar.gz"
    )
    print(f"[checkpoint] Compressing {directory.name}...", flush=True)
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    with tarfile.open(local_archive, "w:gz", compresslevel=1) as bundle:
        for number, path in enumerate(files, start=1):
            if number == 1 or number % 10 == 0 or number == len(files):
                print(
                    f"[checkpoint compress] {number:,}/{len(files):,}: "
                    f"{path.name}",
                    flush=True,
                )
            bundle.add(
                path,
                arcname=str(Path(directory.name) / path.relative_to(directory)),
            )
    partial = archive.with_name(f".{archive.name}.partial")
    try:
        copied = 0
        total = local_archive.stat().st_size
        next_progress = 256 * 1024 * 1024
        with local_archive.open("rb") as source, partial.open("wb") as output:
            for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
                output.write(chunk)
                copied += len(chunk)
                if copied >= next_progress or copied == total:
                    print(
                        f"[checkpoint upload] {copied / 1024**2:.0f}/"
                        f"{total / 1024**2:.0f} MB",
                        flush=True,
                    )
                    next_progress += 256 * 1024 * 1024
        partial.replace(archive)
    finally:
        local_archive.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    print(f"[checkpoint] Saved {archive}", flush=True)


def verify_submission(submission: Path) -> None:
    required = {
        "main.py",
        "artifacts/models/promoted_worker_bc.npz",
        "src/kaggriculture/learning/behavior_model.py",
    }
    with tarfile.open(submission, "r:gz") as archive:
        names = set(archive.getnames())
        missing = required - names
        if missing:
            raise RuntimeError(f"Submission is missing: {sorted(missing)}")
        with tempfile.TemporaryDirectory() as directory:
            archive.extractall(directory, filter="data")
            smoke_code = (
                "from kaggle_environments import make; "
                "env=make('kaggriculture', configuration={'episodeSteps': 24}, "
                "debug=True); "
                "env.run(['main.py', 'random']); "
                "final=env.steps[-1]; "
                "assert all(str(s.status) == 'DONE' for s in final), final; "
                "print([(str(s.status), s.reward) for s in final])"
            )
            run([sys.executable, "-c", smoke_code], cwd=Path(directory))
    print("[submission] Archive and isolated smoke test passed", flush=True)


def submission_fingerprint(path: Path) -> str:
    """Hash archive payloads without timestamps or other tar metadata."""

    digest = hashlib.sha256()
    with tarfile.open(path, "r:gz") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            if not member.isfile():
                continue
            digest.update(member.name.encode("utf-8"))
            digest.update(b"\0")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read {member.name} from {path}")
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def submit_once(
    submission: Path,
    receipt: Path,
    competition: str,
    message: str,
    force: bool,
) -> None:
    checksum = submission_fingerprint(submission)
    if receipt.is_file() and not force:
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous.get("payload_sha256") == checksum:
            print(
                "[submit] This exact archive was already submitted; skipping.",
                flush=True,
            )
            return
    command = [
        "kaggle",
        "competitions",
        "submit",
        competition,
        "-f",
        str(submission),
        "-m",
        message,
    ]
    print(f"\n$ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    output = (completed.stdout + completed.stderr).strip()
    print(output, flush=True)
    receipt.write_text(
        json.dumps(
            {
                "payload_sha256": checksum,
                "submission": str(submission),
                "message": message,
                "kaggle_output": output,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-players", type=int, default=10)
    parser.add_argument("--replays-per-player", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--competition", default="kaggriculture")
    parser.add_argument("--work-root", type=Path, default=Path("/content/kaggriculture_work"))
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--all-submissions", action="store_true")
    parser.add_argument("--both-players", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--force-submit", action="store_true")
    parser.add_argument("--message", default="Teacher worker behavior cloning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_only = not args.all_submissions
    winner_only = not args.both_players
    dataset_id = (
        f"top{args.top_players}_replays{args.replays_per_player}_"
        f"best{int(best_only)}_winner{int(winner_only)}"
    )
    run_id = f"{dataset_id}_epochs{args.epochs}_batch{args.batch_size}"
    work_directory = args.work_root / dataset_id
    replay_directory = work_directory / "teacher_replays"
    dataset_directory = work_directory / "teacher_processed"
    checkpoint_directory = args.drive_root / "checkpoints"
    result_directory = args.drive_root / "results" / run_id
    replay_checkpoint = checkpoint_directory / f"{dataset_id}_replays.tar.gz"
    dataset_checkpoint = checkpoint_directory / f"{dataset_id}_processed.tar.gz"
    result_directory.mkdir(parents=True, exist_ok=True)

    restored_replays = False
    if not args.no_checkpoints:
        restored_replays = restore_checkpoint(
            replay_checkpoint, replay_directory
        )
    replay_directory.mkdir(parents=True, exist_ok=True)
    download_marker = work_directory / "download_complete.json"
    if restored_replays:
        download_marker.write_text("{}\n", encoding="utf-8")
    if not download_marker.is_file():
        download_command = [
            sys.executable,
            "-u",
            "download_top_replays.py",
            "--top-players",
            str(args.top_players),
            "--max-replays-per-player",
            str(args.replays_per_player),
            "--replays",
            str(replay_directory),
        ]
        if best_only:
            download_command.append("--best-submission-only")
        run(download_command)
        download_marker.write_text("{}\n", encoding="utf-8")
    else:
        print(
            f"[download] Reusing {len(list(replay_directory.glob('*.json'))):,} replays",
            flush=True,
        )
    if not args.no_checkpoints:
        save_checkpoint(replay_directory, replay_checkpoint)

    if not complete_dataset(dataset_directory) and not args.no_checkpoints:
        restore_checkpoint(dataset_checkpoint, dataset_directory)
    if not complete_dataset(dataset_directory):
        build_command = [
            sys.executable,
            "-u",
            "build_teacher_dataset.py",
            "--replays",
            str(replay_directory),
            "--output",
            str(dataset_directory),
            "--worker-only",
        ]
        if winner_only:
            build_command.append("--winner-only")
        run(build_command)
    else:
        print(f"[dataset] Reusing {dataset_directory}", flush=True)
    if not args.no_checkpoints:
        save_checkpoint(dataset_directory, dataset_checkpoint)

    model = result_directory / "teacher_worker_bc.npz"
    report = result_directory / "teacher_worker_bc_report.json"
    policy_report = result_directory / "teacher_worker_bc_policy_report.json"
    if not all(path.is_file() for path in (model, report, policy_report)):
        run(
            [
                sys.executable,
                "-u",
                "-m",
                "src.kaggriculture.learning.train_behavior_cloning",
                "--dataset",
                str(dataset_directory / "worker_dataset.jsonl.gz"),
                "--manifest",
                str(dataset_directory / "worker_manifest.json"),
                "--transitions",
                str(dataset_directory / "transitions.jsonl.gz"),
                "--model",
                str(model),
                "--report",
                str(report),
                "--policy-report",
                str(policy_report),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
            ]
        )
    else:
        print(f"[train] Reusing trained model {model}", flush=True)

    submission = result_directory / "submission.tar.gz"
    run(
        [
            sys.executable,
            "package_submission.py",
            "--worker-model",
            str(model),
            "--output",
            str(submission),
        ]
    )
    verify_submission(submission)

    receipt = result_directory / "submission_receipt.json"
    if args.submit:
        submit_once(
            submission,
            receipt,
            args.competition,
            args.message,
            args.force_submit,
        )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "model": str(model),
                "report": str(report),
                "policy_report": str(policy_report),
                "submission": str(submission),
                "submitted": args.submit,
                "receipt": str(receipt) if receipt.is_file() else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
