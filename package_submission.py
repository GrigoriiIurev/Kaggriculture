"""Build a clean Kaggle archive with main.py at its root."""

from __future__ import annotations

import tarfile
from pathlib import Path


OUTPUT = Path("artifacts/submission.tar.gz")
INCLUDED = (
    Path("main.py"),
    Path("src/__init__.py"),
    Path("src/kaggriculture/__init__.py"),
    Path("src/kaggriculture/agent.py"),
    Path("src/kaggriculture/core"),
    Path("src/kaggriculture/planning"),
    Path("artifacts/models/promoted_economic_config.json"),
)


def _exclude_generated(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if "__pycache__" in parts or info.name.endswith((".pyc", ".pyo")):
        return None
    return info


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUTPUT, "w:gz") as archive:
        for path in INCLUDED:
            if path.exists():
                archive.add(path, arcname=str(path), filter=_exclude_generated)
    with tarfile.open(OUTPUT, "r:gz") as archive:
        names = archive.getnames()
    if "main.py" not in names:
        raise RuntimeError("Submission is missing main.py")
    print(f"Created {OUTPUT} with {len(names)} files")


if __name__ == "__main__":
    main()
