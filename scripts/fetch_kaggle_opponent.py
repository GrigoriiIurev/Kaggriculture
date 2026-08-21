"""Download a public Kaggle notebook and extract its generated main.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_KERNEL = "boatlee/v16-rc5-high-score-8c-4s-premium-market-lead"


def extract_written_file(
    notebook_path: str | Path,
    output_path: str | Path,
    filename: str = "main.py",
) -> dict[str, Any]:
    """Extract the body of a ``%%writefile <filename>`` notebook cell."""

    notebook_path = Path(notebook_path)
    output_path = Path(output_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    marker = f"%%writefile {filename}"
    matches: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        lines = source.splitlines(keepends=True)
        if lines and lines[0].strip() == marker:
            matches.append("".join(lines[1:]))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {marker!r} cell, found {len(matches)}"
        )

    payload = matches[0]
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    expected = (
        notebook.get("metadata", {})
        .get("v16_rc5", {})
        .get("agent_sha256")
    )
    if expected and digest != expected:
        raise ValueError(
            "Extracted agent hash does not match notebook metadata: "
            f"expected {expected}, got {digest}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    return {
        "notebook": str(notebook_path),
        "output": str(output_path),
        "sha256": digest,
        "expected_sha256": expected,
        "bytes": output_path.stat().st_size,
    }


def fetch_opponent(kernel: str, output_path: str | Path) -> dict[str, Any]:
    """Pull ``kernel`` with the authenticated Kaggle CLI and extract main.py."""

    with tempfile.TemporaryDirectory(prefix="kaggriculture-opponent-") as tmp:
        directory = Path(tmp)
        print(f"[opponent] Downloading {kernel}...", flush=True)
        subprocess.run(
            ["kaggle", "kernels", "pull", kernel, "-p", str(directory), "-m"],
            check=True,
        )
        notebooks = sorted(directory.glob("*.ipynb"))
        if len(notebooks) != 1:
            raise RuntimeError(
                f"Expected one downloaded notebook, found {len(notebooks)}"
            )
        result = extract_written_file(notebooks[0], output_path)
    print(
        f"[opponent] Extracted {result['bytes']:,} bytes to {result['output']} "
        f"(sha256={result['sha256']})",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default=DEFAULT_KERNEL)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/opponents/boatlee_v16.py")
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        help="Extract a local notebook instead of downloading it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.notebook:
        result = extract_written_file(args.notebook, args.output)
        print(json.dumps(result, indent=2), flush=True)
    else:
        fetch_opponent(args.kernel, args.output)


if __name__ == "__main__":
    main()
