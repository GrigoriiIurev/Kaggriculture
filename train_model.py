"""Optimize production policy parameters through complete local games."""

from __future__ import annotations

import argparse
import json

from src.kaggriculture.learning.policy_search import run_search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-games", type=int, default=2)
    parser.add_argument("--confirmation-games", type=int, default=8)
    parser.add_argument("--validation-games", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=2)
    args = parser.parse_args()
    report = run_search(
        screening_games=args.screening_games,
        confirmation_games=args.confirmation_games,
        validation_games=args.validation_games,
        finalists=args.finalists,
    )
    summary = {
        "winner": report["winner"],
        "promoted": report["promoted"],
        "final_validation": report["final_validation"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
