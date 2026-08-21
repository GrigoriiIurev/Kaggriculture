"""Build a clean Kaggle archive with main.py at its root."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


OUTPUT = Path("artifacts/submission.tar.gz")
BASE_INCLUDED = (
    Path("main.py"),
    Path("src/__init__.py"),
    Path("src/kaggriculture/__init__.py"),
    Path("src/kaggriculture/agent.py"),
    Path("src/kaggriculture/core"),
    Path("src/kaggriculture/planning"),
    Path("artifacts/models/promoted_economic_config.json"),
)
WORKER_MODEL_INCLUDED = (
    Path("src/kaggriculture/data/__init__.py"),
    Path("src/kaggriculture/data/feature_extractor.py"),
    Path("src/kaggriculture/data/worker_dataset.py"),
    Path("src/kaggriculture/learning/__init__.py"),
    Path("src/kaggriculture/learning/behavior_model.py"),
)
WORKER_MODEL_ARCHIVE_PATH = "artifacts/models/promoted_worker_bc.npz"


def _exclude_generated(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if "__pycache__" in parts or info.name.endswith((".pyc", ".pyo")):
        return None
    return info


def build_submission(
    output_path: str | Path = OUTPUT,
    worker_model_path: str | Path | None = None,
) -> dict[str, object]:
    output_path = Path(output_path)
    worker_model = Path(worker_model_path) if worker_model_path else None
    if worker_model is not None and not worker_model.is_file():
        raise FileNotFoundError(f"Worker model not found: {worker_model}")

    included = list(BASE_INCLUDED)
    if worker_model is not None:
        included.extend(WORKER_MODEL_INCLUDED)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(
        output_path, "w:gz", format=tarfile.GNU_FORMAT
    ) as archive:
        for path in included:
            if path.exists():
                archive.add(path, arcname=str(path), filter=_exclude_generated)
        if worker_model is not None:
            archive.add(worker_model, arcname=WORKER_MODEL_ARCHIVE_PATH)
    with tarfile.open(output_path, "r:gz") as archive:
        names = archive.getnames()
    if "main.py" not in names:
        raise RuntimeError("Submission is missing main.py")
    if worker_model is not None and WORKER_MODEL_ARCHIVE_PATH not in names:
        raise RuntimeError("Submission is missing the worker model")
    return {
        "output": str(output_path),
        "files": len(names),
        "bytes": output_path.stat().st_size,
        "agent": "behavior_cloning_workers" if worker_model else "rule_based",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--worker-model",
        type=Path,
        help="Exported worker .npz to embed in the hybrid submission",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_submission(args.output, args.worker_model)
    size_mb = int(result["bytes"]) / 1024 / 1024
    print(
        f"Created {result['output']} with {result['files']} files "
        f"({size_mb:.2f} MB, {result['agent']})"
    )


if __name__ == "__main__":
    main()
