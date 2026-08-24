"""Fetch public Kaggle notebooks and recover the agent they package."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NotebookAgent:
    slug: str
    kernel: str
    title: str
    role: str
    expected_sha256: str


DEFAULT_NOTEBOOKS = (
    NotebookAgent(
        "deniz_v111",
        "denizeryilmaz/v111-8c4s-economic-core-premium-lead",
        "V111 8C4S Economic Core Premium Lead",
        "economic_core",
        "f029fa0cb66a9eb509afbe44e3f59b800332d0419db91607183410e4089c4d19",
    ),
    NotebookAgent(
        "ray_c95",
        "raykkretzschmar/kaggriculture-findings-from-zero-to-top-meta",
        "Findings from Zero to Top Meta (latest C95 artifact)",
        "adaptive_market",
        "489f5d197527f107027626cce79d850fd2ca90edd43d94384b849b6511e27bdb",
    ),
    NotebookAgent(
        "boatlee_r5a",
        "boatlee/v16-rc5-r5a-high-score-8c-4s-recovery",
        "V16 RC5 R5A Recovery",
        "recovery",
        "7f87c941af3050d0f21376f2843b324d7a06a1a8c050fa554cf07a769e5c937c",
    ),
    NotebookAgent(
        "kaito_v43",
        "kaitofukami/103-128-fresh-public-v43-sparse-shop-hybrid",
        "V43 Sparse Shop Hybrid",
        "daily_market_hybrid",
        "69f06a802b62aa08f28705dab5728eb924bb6a7c23ffe0164f65b104cc3dadf3",
    ),
    NotebookAgent(
        "bruce_route1",
        "bruceqdu/my-2026-08-04-high-score-pipeline",
        "2026-08-04 High-Score Route 1",
        "alternative_route",
        "ed8c8420514acb5a96c0d44cfd42a8786e49c7cdc01a0de61d2e6b8997dda87a",
    ),
    NotebookAgent(
        "ray_k320",
        "raykkretzschmar/kaggriculture-rank-your-agent",
        "K320 Adaptive Rank-1",
        "rank_ladder_champion",
        "6c709f6d3ce6cf221a9495de7e716fcd1b660e3bbc8ee5679b63233d0265a812",
    ),
)


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _valid_agent_source(payload: bytes) -> bool:
    try:
        text = payload.decode("utf-8")
        tree = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError):
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "agent"
        for node in ast.walk(tree)
    )


def _assigned_strings(source: str) -> list[tuple[str, bytes]]:
    """Read literal strings without executing arbitrary notebook code."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found: list[tuple[str, bytes]] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        value_node = node.value
        if not isinstance(target, ast.Name) or value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError, SyntaxError):
            continue
        if isinstance(value, str):
            found.append((target.id, value.encode("utf-8")))
        elif isinstance(value, bytes):
            found.append((target.id, value))
        elif isinstance(value, (list, tuple)) and value and all(
            isinstance(part, str) for part in value
        ):
            found.append((target.id, "".join(value).encode("utf-8")))
    return found


def _decoded_candidates(payload: bytes) -> list[bytes]:
    candidates = [payload]
    for decoder in (base64.b85decode, base64.b64decode):
        try:
            decoded = decoder(payload)
        except (ValueError, base64.binascii.Error):
            continue
        candidates.append(decoded)
        try:
            candidates.append(zlib.decompress(decoded))
        except zlib.error:
            pass
    return candidates


def extract_agent_source(notebook: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """Return the last packaged main.py from common public-notebook formats."""

    candidates: list[tuple[int, str, bytes]] = []
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        lines = source.splitlines(keepends=True)
        if lines and lines[0].strip() in {
            "%%writefile main.py",
            "%%writefile /kaggle/working/main.py",
        }:
            payload = "".join(lines[1:]).encode("utf-8")
            if _valid_agent_source(payload):
                candidates.append((cell_index, "writefile", payload))
        for variable, literal in _assigned_strings(source):
            for payload in _decoded_candidates(literal):
                if _valid_agent_source(payload):
                    candidates.append((cell_index, variable, payload))
    if not candidates:
        raise ValueError("Notebook does not contain a recoverable main.py agent")

    cell_index, method, payload = candidates[-1]
    return payload, {
        "cell_index": cell_index,
        "method": method,
        "candidate_count": len(candidates),
    }


def fetch_notebook_json(
    kernel: str,
    *,
    timeout: int = 90,
    kaggle_executable: str | None = None,
) -> dict[str, Any]:
    """Fetch public source via the API, with Kaggle CLI as a fallback."""

    url = f"https://www.kaggle.com/api/v1/kernels/pull/{kernel}"
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "Kaggriculture-League/1.0"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            envelope = json.load(response)
        source = envelope["blob"]["source"]
        return json.loads(source) if isinstance(source, str) else source
    except Exception as api_error:
        sibling = Path(sys.executable).with_name("kaggle")
        executable = (
            kaggle_executable
            or shutil.which("kaggle")
            or (str(sibling) if sibling.is_file() else None)
        )
        if not executable:
            raise RuntimeError(
                f"Could not fetch {kernel} from the public API and Kaggle CLI is absent"
            ) from api_error
        with tempfile.TemporaryDirectory(prefix="kaggriculture-league-") as tmp:
            subprocess.run(
                [executable, "kernels", "pull", kernel, "-p", tmp, "-m"],
                check=True,
                timeout=timeout,
            )
            notebooks = sorted(Path(tmp).glob("*.ipynb"))
            if len(notebooks) != 1:
                raise RuntimeError(
                    f"Expected one notebook for {kernel}, found {len(notebooks)}"
                ) from api_error
            return json.loads(notebooks[0].read_text(encoding="utf-8"))


def fetch_notebook_agents(
    output_directory: str | Path,
    *,
    refresh: bool = False,
    timeout: int = 90,
    kaggle_executable: str | None = None,
    sources: tuple[NotebookAgent, ...] = DEFAULT_NOTEBOOKS,
) -> dict[str, Any]:
    """Materialize the configured public opponents and a reproducible manifest."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        destination = output_directory / f"{source.slug}.py"
        reused = destination.is_file() and not refresh
        extraction: dict[str, Any] = {"method": "cached"}
        if not reused:
            print(
                f"[fetch {index}/{len(sources)}] {source.kernel}",
                flush=True,
            )
            notebook = fetch_notebook_json(
                source.kernel,
                timeout=timeout,
                kaggle_executable=kaggle_executable,
            )
            payload, extraction = extract_agent_source(notebook)
            destination.write_bytes(payload)
        payload = destination.read_bytes()
        if not _valid_agent_source(payload):
            raise ValueError(f"Extracted file is not an agent: {destination}")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != source.expected_sha256:
            raise ValueError(
                f"Agent hash changed for {source.kernel}: expected "
                f"{source.expected_sha256}, got {digest}"
            )
        record = {
            **asdict(source),
            "path": str(destination),
            "sha256": digest,
            "bytes": len(payload),
            "reused": reused,
            "extraction": extraction,
        }
        records.append(record)
        print(
            f"[fetch {index}/{len(sources)}] ready {source.slug}: "
            f"{len(payload):,} bytes",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "agents": records,
        "source_count": len(records),
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
