"""Build Replay Warehouse datasets and a diagnostic report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.kaggriculture.analysis.replay_warehouse import build_replay_warehouse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--team")
    parser.add_argument("--episode-index", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_replay_warehouse(
        replay_directory=args.replays,
        output_directory=args.output,
        team_name=args.team,
        episode_index_path=args.episode_index,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
