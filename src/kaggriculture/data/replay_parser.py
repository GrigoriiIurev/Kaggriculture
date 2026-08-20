"""Read Kaggriculture replay files and recover agent decision transitions.

Kaggle stores the action chosen from frame ``t`` in frame ``t + 1``.  A
training transition is therefore ``steps[t].observation ->
steps[t + 1].action -> steps[t + 1].observation``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ReplayParseError(ValueError):
    """Raised when a replay does not have the expected Kaggle structure."""


@dataclass(frozen=True)
class Replay:
    source_path: Path
    episode_id: int
    teams: tuple[str, ...]
    rewards: tuple[float, ...]
    statuses: tuple[str, ...]
    configuration: dict[str, Any]
    steps: list[list[dict[str, Any]]]

    @property
    def turns_per_day(self) -> int:
        return int(self.configuration.get("turnsPerDay", 24))

    @property
    def is_self_play(self) -> bool:
        return len(set(self.teams)) < len(self.teams)


@dataclass(frozen=True)
class Transition:
    episode_id: int
    episode_type: str
    seat: int
    team: str
    opponent: str
    step: int
    observation: dict[str, Any]
    action: dict[str, Any]
    next_observation: dict[str, Any]
    final_reward: float
    opponent_final_reward: float
    margin: float
    outcome: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "episode_type": self.episode_type,
            "seat": self.seat,
            "team": self.team,
            "opponent": self.opponent,
            "step": self.step,
            "observation": self.observation,
            "action": self.action,
            "next_observation": self.next_observation,
            "final_reward": self.final_reward,
            "opponent_final_reward": self.opponent_final_reward,
            "margin": self.margin,
            "outcome": self.outcome,
        }


def load_replay(path: str | Path) -> Replay:
    """Load and minimally validate one two-player replay."""

    source_path = Path(path)
    try:
        with source_path.open(encoding="utf-8") as replay_file:
            raw = json.load(replay_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayParseError(f"Cannot read {source_path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ReplayParseError(f"Replay root must be an object: {source_path}")

    info = raw.get("info")
    steps = raw.get("steps")
    rewards = raw.get("rewards")
    statuses = raw.get("statuses")
    configuration = raw.get("configuration", {})
    if not isinstance(info, Mapping) or not isinstance(steps, list) or len(steps) < 2:
        raise ReplayParseError(f"Replay has no usable info/steps: {source_path}")
    if not isinstance(configuration, Mapping):
        raise ReplayParseError(f"Replay configuration must be an object: {source_path}")

    teams = info.get("TeamNames")
    episode_id = info.get("EpisodeId", raw.get("id"))
    if not isinstance(teams, Sequence) or isinstance(teams, (str, bytes)) or len(teams) != 2:
        raise ReplayParseError(f"Expected exactly two TeamNames: {source_path}")
    if not isinstance(rewards, Sequence) or len(rewards) != 2:
        raise ReplayParseError(f"Expected exactly two rewards: {source_path}")
    if not isinstance(statuses, Sequence) or len(statuses) != 2:
        raise ReplayParseError(f"Expected exactly two statuses: {source_path}")

    normalized_steps: list[list[dict[str, Any]]] = []
    for frame_index, frame in enumerate(steps):
        if not isinstance(frame, list) or len(frame) != 2:
            raise ReplayParseError(
                f"Frame {frame_index} must contain two player states: {source_path}"
            )
        if any(not isinstance(player_state, dict) for player_state in frame):
            raise ReplayParseError(f"Frame {frame_index} contains an invalid state: {source_path}")
        normalized_steps.append(frame)

    try:
        normalized_rewards = tuple(float(reward) for reward in rewards)
        normalized_id = int(episode_id)
    except (TypeError, ValueError) as exc:
        raise ReplayParseError(f"Invalid episode id or rewards: {source_path}") from exc

    return Replay(
        source_path=source_path,
        episode_id=normalized_id,
        teams=tuple(str(team) for team in teams),
        rewards=normalized_rewards,
        statuses=tuple(str(status) for status in statuses),
        configuration=dict(configuration),
        steps=normalized_steps,
    )


def normalize_observation(
    observation: Mapping[str, Any], turns_per_day: int, frame_index: int
) -> dict[str, Any]:
    """Remove runtime noise and restore the omitted step for the second seat."""

    normalized = {
        key: value for key, value in observation.items() if key != "remainingOverageTime"
    }
    if "step" not in normalized:
        try:
            normalized["step"] = int(normalized["day"]) * turns_per_day + int(
                normalized["hour"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplayParseError(
                f"Cannot derive step for frame {frame_index}: missing day/hour"
            ) from exc
    normalized["step"] = int(normalized["step"])
    return normalized


def seats_for_team(replay: Replay, team_name: str | None) -> tuple[int, ...]:
    if team_name is None:
        return tuple(range(len(replay.teams)))
    return tuple(index for index, name in enumerate(replay.teams) if name == team_name)


def iter_transitions(replay: Replay, team_name: str | None = None) -> Iterator[Transition]:
    """Yield correctly aligned decisions for matching seats.

    Passing no team name intentionally returns both players.  Dataset builders
    should pass a team name to avoid mixing an opponent's policy into the data.
    """

    for seat in seats_for_team(replay, team_name):
        opponent_seat = 1 - seat
        final_reward = replay.rewards[seat]
        opponent_reward = replay.rewards[opponent_seat]
        margin = final_reward - opponent_reward
        outcome = "win" if margin > 0 else "loss" if margin < 0 else "tie"
        episode_type = "validation" if replay.is_self_play else "public"

        for frame_index in range(len(replay.steps) - 1):
            current = replay.steps[frame_index][seat]
            following = replay.steps[frame_index + 1][seat]
            observation = current.get("observation")
            next_observation = following.get("observation")
            action = following.get("action")
            if not isinstance(observation, Mapping) or not isinstance(
                next_observation, Mapping
            ):
                raise ReplayParseError(
                    f"Missing observation in episode {replay.episode_id}, frame {frame_index}"
                )
            if not isinstance(action, Mapping):
                raise ReplayParseError(
                    f"Missing action in episode {replay.episode_id}, frame {frame_index + 1}"
                )

            clean_observation = normalize_observation(
                observation, replay.turns_per_day, frame_index
            )
            clean_next = normalize_observation(
                next_observation, replay.turns_per_day, frame_index + 1
            )
            yield Transition(
                episode_id=replay.episode_id,
                episode_type=episode_type,
                seat=seat,
                team=replay.teams[seat],
                opponent=replay.teams[opponent_seat],
                step=clean_observation["step"],
                observation=clean_observation,
                action=dict(action),
                next_observation=clean_next,
                final_reward=final_reward,
                opponent_final_reward=opponent_reward,
                margin=margin,
                outcome=outcome,
            )


def infer_team_name(paths: Iterable[str | Path]) -> str:
    """Infer the submitted team as the name occurring in the most replays."""

    counts: Counter[str] = Counter()
    for path in paths:
        counts.update(load_replay(path).teams)
    if not counts:
        raise ReplayParseError("No replay files found")
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise ReplayParseError("Cannot infer team name: the most common names are tied")
    return ranked[0][0]
