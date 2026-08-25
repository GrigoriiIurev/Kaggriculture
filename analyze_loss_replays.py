"""Build detailed loss diagnostics from an existing Replay Warehouse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.kaggriculture.analysis.loss_replay_analyzer import (
    build_loss_replay_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--team")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.analysis / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Replay Warehouse manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    team = args.team or str(manifest["team_name"])
    report = build_loss_replay_analysis(
        args.analysis / manifest["files"]["daily_macro"],
        args.analysis,
        team,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
