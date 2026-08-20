"""One reproducible pipeline for every replay-derived dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .economic_dataset import build_economic_dataset
from .economic_value_dataset import build_value_dataset
from .feature_extractor import build_feature_dataset
from .outcome_logger import build_dataset as build_replay_dataset
from .worker_dataset import build_worker_dataset


def _run_stage(name: str, function):
    print(f"[{name}] started", flush=True)
    result = function()
    print(f"[{name}] complete", flush=True)
    return result


def build_all_datasets(
    replay_directory: str | Path = "data/replays",
    dataset_directory: str | Path = "data/processed",
    team: str = "Grigorii IU",
) -> dict[str, Any]:
    replay_directory = Path(replay_directory)
    dataset_directory = Path(dataset_directory)
    dataset_directory.mkdir(parents=True, exist_ok=True)

    replay = _run_stage(
        "1/5 transitions",
        lambda: build_replay_dataset(replay_directory, dataset_directory, team),
    )
    features = _run_stage(
        "2/5 features",
        lambda: build_feature_dataset(
            dataset_directory / "transitions.jsonl.gz",
            dataset_directory / "features.jsonl.gz",
        ),
    )
    workers = _run_stage(
        "3/5 workers",
        lambda: build_worker_dataset(
            dataset_directory / "transitions.jsonl.gz",
            dataset_directory / "worker_dataset.jsonl.gz",
        ),
    )
    economics = _run_stage(
        "4/5 economics",
        lambda: build_economic_dataset(
            dataset_directory / "features.jsonl.gz",
            dataset_directory / "economic_dataset.jsonl.gz",
        ),
    )
    value = _run_stage(
        "5/5 value",
        lambda: build_value_dataset(
            dataset_directory / "economic_dataset.jsonl.gz",
            dataset_directory / "economic_value_dataset.jsonl.gz",
        ),
    )
    return {
        "replays": replay,
        "features": features,
        "workers": workers,
        "economics": economics,
        "value": value,
    }
