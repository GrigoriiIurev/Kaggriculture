"""Package the Stage 3 residual together with its immutable Stage 2 incumbent."""

from __future__ import annotations

import argparse
import tarfile
import tempfile
from pathlib import Path


LEAGUE_MODEL_ARCHIVE_PATH = "artifacts/league/league_policy.npz"
INCUMBENT_MODEL_ARCHIVE_PATH = "artifacts/league/incumbent_meta_policy.npz"
EXPERT_ARCHIVE_PATH = "artifacts/league/expert_agent.py"
ENTRYPOINT = '''"""Kaggle entry point for the Stage 3 league residual agent."""
from pathlib import Path

from src.kaggriculture.rl import league_policy


ROOT = Path(league_policy.__file__).resolve().parents[3]
agent = league_policy.LeagueResidualAgent(
    ROOT / "artifacts/league/league_policy.npz",
    ROOT / "artifacts/league/incumbent_meta_policy.npz",
    ROOT / "artifacts/league/expert_agent.py",
)
'''

INCLUDED = (
    Path("src/__init__.py"),
    Path("src/kaggriculture/__init__.py"),
    Path("src/kaggriculture/agent.py"),
    Path("src/kaggriculture/core"),
    Path("src/kaggriculture/planning"),
    Path("src/kaggriculture/data/__init__.py"),
    Path("src/kaggriculture/data/feature_extractor.py"),
    Path("src/kaggriculture/rl/__init__.py"),
    Path("src/kaggriculture/rl/meta_policy.py"),
    Path("src/kaggriculture/rl/league_policy.py"),
    Path("artifacts/models/promoted_economic_config.json"),
)


def _exclude_generated(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if "__pycache__" in parts or info.name.endswith((".pyc", ".pyo")):
        return None
    return info


def build_league_submission(
    output_path: str | Path,
    league_model_path: str | Path,
    incumbent_model_path: str | Path,
    expert_path: str | Path,
) -> dict[str, object]:
    output_path = Path(output_path)
    files = {
        LEAGUE_MODEL_ARCHIVE_PATH: Path(league_model_path),
        INCUMBENT_MODEL_ARCHIVE_PATH: Path(incumbent_model_path),
        EXPERT_ARCHIVE_PATH: Path(expert_path),
    }
    for label, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kaggriculture-league-entry-") as directory:
        entrypoint = Path(directory) / "main.py"
        entrypoint.write_text(ENTRYPOINT, encoding="utf-8")
        with tarfile.open(output_path, "w:gz", format=tarfile.GNU_FORMAT) as archive:
            archive.add(entrypoint, arcname="main.py")
            for path in INCLUDED:
                if path.exists():
                    archive.add(path, arcname=str(path), filter=_exclude_generated)
            for archive_name, source in files.items():
                archive.add(source, arcname=archive_name)

    with output_path.open("rb") as stream:
        if stream.read(2) != b"\x1f\x8b":
            raise RuntimeError("Output is not gzip compressed")
    with tarfile.open(output_path, "r:gz") as archive:
        names = set(archive.getnames())
    required = {"main.py", *files}
    missing = required - names
    if missing:
        raise RuntimeError(f"Submission is missing {sorted(missing)}")
    return {
        "output": str(output_path),
        "bytes": output_path.stat().st_size,
        "files": len(names),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--league-model", type=Path, required=True)
    parser.add_argument("--incumbent-model", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_league_submission(
        args.output, args.league_model, args.incumbent_model, args.expert
    )
    print(
        f"Created {result['output']} with {result['files']} files "
        f"({result['bytes'] / 1024 / 1024:.2f} MB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
