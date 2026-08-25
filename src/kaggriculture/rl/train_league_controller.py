"""Train a conservative market residual against the Stage 2 opponent league."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .league_policy import (
    ACTION_DIMS,
    ACTION_NAMES,
    LEAGUE_POLICY_VERSION,
    RESIDUAL_PRODUCTS,
    MarketHistoryFeatures,
    NumpyLeaguePolicy,
)
from .league_training_env import KaggricultureLeagueEnv


def save_fallback_policy(
    path: str | Path,
    feature_count: int = MarketHistoryFeatures.feature_count,
    hidden_sizes: tuple[int, int] = (128, 64),
) -> None:
    """Create a valid policy that leaves every incumbent action unchanged."""

    first, second = hidden_sizes
    bias = np.zeros(sum(ACTION_DIMS), dtype=np.float32)
    offset = 0
    for size in ACTION_DIMS:
        bias[offset] = 1.0
        offset += size
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        w0=np.zeros((first, feature_count), dtype=np.float32),
        b0=np.zeros(first, dtype=np.float32),
        w1=np.zeros((second, first), dtype=np.float32),
        b1=np.zeros(second, dtype=np.float32),
        w2=np.zeros((sum(ACTION_DIMS), second), dtype=np.float32),
        b2=bias,
        feature_count=np.asarray(feature_count),
        action_dims=np.asarray(ACTION_DIMS),
        policy_version=np.asarray(LEAGUE_POLICY_VERSION),
    )


def export_sb3_policy(model: Any, path: str | Path) -> None:
    """Export the PPO MultiDiscrete actor for dependency-free NumPy inference."""

    import torch

    hidden = [
        layer
        for layer in model.policy.mlp_extractor.policy_net
        if isinstance(layer, torch.nn.Linear)
    ]
    layers = [*hidden, model.policy.action_net]
    if len(layers) != 3:
        raise ValueError(f"Expected 3 actor linear layers, found {len(layers)}")
    arrays: dict[str, Any] = {
        "feature_count": np.asarray(model.observation_space.shape[0]),
        "action_dims": np.asarray(ACTION_DIMS),
        "policy_version": np.asarray(LEAGUE_POLICY_VERSION),
    }
    for index, layer in enumerate(layers):
        arrays[f"w{index}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{index}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path, **arrays)


def _load_pool(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    opponents = payload.get("training_opponents", [])
    if not opponents:
        raise ValueError(f"No training_opponents in {path}")
    missing = [row["path"] for row in opponents if not Path(row["path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Opponent files are missing: {missing[:3]}")
    veto = [str(value) for value in payload.get("veto_opponents", [])]
    return opponents, veto


def evaluate_policy(
    selector: Callable[[np.ndarray], Sequence[int]],
    incumbent_path: Path,
    opponents: list[dict[str, Any]],
    *,
    seeds: Sequence[int],
    episode_steps: int,
) -> dict[str, Any]:
    """Evaluate every opponent on identical seeds and both seats."""

    rows: list[dict[str, Any]] = []
    total_choice_counts = [
        [0 for _ in ACTION_NAMES] for _ in RESIDUAL_PRODUCTS
    ]
    total_decisions = total_effective = 0
    for opponent_index, opponent in enumerate(opponents):
        wins = ties = losses = errors = 0
        margins: list[float] = []
        for seed in seeds:
            for seat in (0, 1):
                environment = KaggricultureLeagueEnv(
                    incumbent_path,
                    [opponent["path"]],
                    episode_steps=episode_steps,
                    seed_offset=0,
                    fixed_opponent=0,
                    fixed_seat=seat,
                )
                try:
                    observation, _ = environment.reset(seed=seed)
                    done = False
                    info: dict[str, Any] = {}
                    next_turn_log = 240
                    while not done:
                        choice = np.asarray(selector(observation), dtype=np.int64)
                        observation, _, done, _, info = environment.step(choice)
                        if "choices" in info:
                            total_decisions += 1
                            total_effective += bool(info["residual_effective"])
                            for product_index, selected in enumerate(info["choices"]):
                                total_choice_counts[product_index][int(selected)] += 1
                        game_turn = int(info.get("game_turn", episode_steps))
                        if game_turn >= next_turn_log and not done:
                            print(
                                f"[evaluate {opponent['slug']}] seed={seed} "
                                f"seat={seat} turn={game_turn}/{episode_steps}",
                                flush=True,
                            )
                            while next_turn_log <= game_turn:
                                next_turn_log += 240
                    outcome = int(info["outcome"])
                    wins += outcome > 0
                    ties += outcome == 0
                    losses += outcome < 0
                    margins.append(float(info["money_margin"]))
                except Exception as exc:
                    errors += 1
                    print(
                        f"[evaluate error] {opponent['slug']} seed={seed} seat={seat}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                print(
                    f"[evaluate {opponent_index + 1}/{len(opponents)}] "
                    f"{opponent['slug']}: {wins}W-{ties}T-{losses}L-{errors}E",
                    flush=True,
                )
        valid = wins + ties + losses
        rows.append(
            {
                "slug": opponent["slug"],
                "veto": bool(opponent.get("veto", False)),
                "games": valid + errors,
                "valid_games": valid,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "errors": errors,
                "score_rate": (wins + 0.5 * ties) / valid if valid else 0.0,
                "mean_money_margin": float(np.mean(margins)) if margins else 0.0,
            }
        )
    valid_games = sum(row["valid_games"] for row in rows)
    wins = sum(row["wins"] for row in rows)
    ties = sum(row["ties"] for row in rows)
    return {
        "games": sum(row["games"] for row in rows),
        "valid_games": valid_games,
        "errors": sum(row["errors"] for row in rows),
        "wins": wins,
        "ties": ties,
        "losses": sum(row["losses"] for row in rows),
        "score_rate": (wins + 0.5 * ties) / valid_games if valid_games else 0.0,
        "mean_money_margin": float(
            np.mean([row["mean_money_margin"] for row in rows])
        ),
        "market_decisions": total_decisions,
        "effective_decisions": total_effective,
        "effective_rate": (
            total_effective / total_decisions if total_decisions else 0.0
        ),
        "choice_counts": {
            product: dict(zip(ACTION_NAMES, counts))
            for product, counts in zip(RESIDUAL_PRODUCTS, total_choice_counts)
        },
        "opponents": rows,
    }


def promotion_gate(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Require overall improvement and prevent meaningful veto regressions."""

    base_by_name = {row["slug"]: row for row in baseline["opponents"]}
    veto_checks = []
    for row in candidate["opponents"]:
        if not row["veto"]:
            continue
        base = base_by_name[row["slug"]]
        score_drop = base["score_rate"] - row["score_rate"]
        margin_drop = base["mean_money_margin"] - row["mean_money_margin"]
        passed = score_drop <= 0.25 and not (
            score_drop > 0 and margin_drop > 5_000
        )
        veto_checks.append(
            {
                "slug": row["slug"],
                "passed": passed,
                "score_rate_change": row["score_rate"] - base["score_rate"],
                "mean_margin_change": row["mean_money_margin"]
                - base["mean_money_margin"],
            }
        )
    ranking = (candidate["score_rate"], candidate["mean_money_margin"])
    baseline_ranking = (baseline["score_rate"], baseline["mean_money_margin"])
    improved = ranking > baseline_ranking
    passed = (
        candidate["errors"] == 0
        and candidate.get("effective_decisions", 0) > 0
        and improved
        and all(check["passed"] for check in veto_checks)
    )
    return {
        "passed": passed,
        "improved_over_incumbent": improved,
        "error_free": candidate["errors"] == 0,
        "changed_market": candidate.get("effective_decisions", 0) > 0,
        "score_rate_change": candidate["score_rate"] - baseline["score_rate"],
        "mean_margin_change": candidate["mean_money_margin"]
        - baseline["mean_money_margin"],
        "veto_checks": veto_checks,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_rounds <= 0 or args.steps_per_round <= 0:
        raise ValueError("Training rounds and timesteps must be positive")
    if args.log_every_steps <= 0 or args.checkpoint_every_steps <= 0:
        raise ValueError("Log and checkpoint intervals must be positive")
    if args.train_envs <= 0 or args.eval_seed_count <= 0:
        raise ValueError("Environment and evaluation counts must be positive")
    if args.weakness_bonus < 0 or args.veto_weight_bonus <= 0:
        raise ValueError("Curriculum weight multipliers must be non-negative")
    if not 0 < args.catastrophic_score_drop <= 1:
        raise ValueError("catastrophic_score_drop must be in (0, 1]")
    print("[setup] Loading league and RL libraries", flush=True)
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:  # pragma: no cover - exercised in Colab
        raise ImportError("Install requirements-rl.txt before training") from exc

    opponents, veto_names = _load_pool(args.opponent_pool)
    for row in opponents:
        row["veto"] = row["slug"] in veto_names or bool(row.get("veto", False))
    paths = [row["path"] for row in opponents]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fallback = args.output_dir / "incumbent_fallback_policy.npz"
    best_numpy = args.output_dir / "best_league_policy.npz"
    last_numpy = args.output_dir / "last_league_policy.npz"
    last_model = args.output_dir / "last_league_ppo_model.zip"
    best_model = args.output_dir / "best_league_ppo_model.zip"
    anchor_model = args.output_dir / "safe_training_anchor.zip"
    report_path = args.output_dir / "league_training_report.json"
    if not fallback.is_file():
        save_fallback_policy(fallback)
    if not best_numpy.is_file():
        shutil.copy2(fallback, best_numpy)

    eval_seeds = tuple(
        range(args.eval_seed_offset, args.eval_seed_offset + args.eval_seed_count)
    )
    print(
        f"[baseline] {len(opponents)} opponents, seeds={list(eval_seeds)}, both seats",
        flush=True,
    )
    baseline = evaluate_policy(
        lambda _: np.zeros(len(ACTION_DIMS), dtype=np.int64),
        args.incumbent,
        opponents,
        seeds=eval_seeds,
        episode_steps=args.episode_steps,
    )
    print(f"[baseline result] {json.dumps(baseline)}", flush=True)

    baseline_by_name = {row["slug"]: row for row in baseline["opponents"]}
    raw_weights = []
    curriculum = {}
    for row in opponents:
        score_rate = baseline_by_name[row["slug"]]["score_rate"]
        multiplier = 1.0 + args.weakness_bonus * (1.0 - score_rate)
        if row["veto"]:
            multiplier *= args.veto_weight_bonus
        weight = float(row.get("sampling_weight", 1.0)) * multiplier
        raw_weights.append(weight)
        curriculum[row["slug"]] = {
            "baseline_score_rate": score_rate,
            "multiplier": multiplier,
            "raw_weight": weight,
        }
    weight_total = sum(raw_weights)
    weights = [weight / weight_total for weight in raw_weights]
    for row, weight in zip(opponents, weights):
        curriculum[row["slug"]]["sampling_weight"] = weight
    print(f"[curriculum] {json.dumps(curriculum)}", flush=True)

    factories = []
    for index in range(args.train_envs):
        def factory(index: int = index) -> KaggricultureLeagueEnv:
            return KaggricultureLeagueEnv(
                args.incumbent,
                paths,
                weights,
                episode_steps=args.episode_steps,
                seed_offset=args.train_seed_offset + index * 1_000_000,
            )
        factories.append(factory)
    vector_environment = DummyVecEnv(factories)
    if last_model.is_file():
        print(f"[train] Resuming {last_model}", flush=True)
        model = PPO.load(last_model, env=vector_environment, device=args.device)
    else:
        model = PPO(
            "MlpPolicy",
            vector_environment,
            policy_kwargs={
                "net_arch": {"pi": [128, 64], "vf": [128, 64]},
                "activation_fn": torch.nn.Tanh,
            },
            learning_rate=args.learning_rate,
            n_steps=args.rollout_steps,
            batch_size=args.batch_size,
            n_epochs=args.ppo_epochs,
            gamma=1.0,
            gae_lambda=1.0,
            ent_coef=args.entropy,
            verbose=1,
            device=args.device,
            seed=args.model_seed,
        )
        with torch.no_grad():
            model.policy.action_net.bias.zero_()
            offset = 0
            for size in ACTION_DIMS:
                model.policy.action_net.bias[offset] = args.incumbent_initial_bias
                offset += size
        print(
            "[train] Initialized an exploratory policy; the promotion gate "
            "still protects the incumbent",
            flush=True,
        )
        model.save(anchor_model)
        print(f"[train] Saved initial safety anchor {anchor_model}", flush=True)
    if not anchor_model.is_file():
        model.save(anchor_model)

    class ProgressCallback(BaseCallback):
        def __init__(self, interval: int, checkpoint_interval: int) -> None:
            super().__init__()
            self.interval = interval
            self.next_log = model.num_timesteps + interval
            self.checkpoint_interval = checkpoint_interval
            self.next_checkpoint = model.num_timesteps + checkpoint_interval
            self.decisions = 0
            self.effective = 0
            self.choice_counts = [
                [0 for _ in ACTION_NAMES] for _ in RESIDUAL_PRODUCTS
            ]

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "choices" not in info:
                    continue
                self.decisions += 1
                self.effective += bool(info.get("residual_effective", False))
                for product_index, choice in enumerate(info["choices"]):
                    self.choice_counts[product_index][int(choice)] += 1
            if self.num_timesteps >= self.next_log:
                print(f"[train progress] {self.num_timesteps:,} timesteps", flush=True)
                print(
                    "[train actions] "
                    f"decisions={self.decisions:,}, effective={self.effective:,} "
                    f"({self.effective / max(1, self.decisions):.1%}), "
                    f"counts={dict(zip(RESIDUAL_PRODUCTS, self.choice_counts))}",
                    flush=True,
                )
                while self.next_log <= self.num_timesteps:
                    self.next_log += self.interval
            if self.num_timesteps >= self.next_checkpoint:
                model.save(last_model)
                print(
                    f"[checkpoint] saved at {self.num_timesteps:,} timesteps",
                    flush=True,
                )
                while self.next_checkpoint <= self.num_timesteps:
                    self.next_checkpoint += self.checkpoint_interval
            return True

    previous = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else {}
    )
    rounds = list(previous.get("rounds", []))
    best_candidate: dict[str, Any] | None = previous.get("best_candidate")
    best_ranking = (
        (best_candidate["score_rate"], best_candidate["mean_money_margin"])
        if best_candidate
        else (-1.0, float("-inf"))
    )
    for round_index in range(args.max_rounds):
        print(
            f"[train round {round_index + 1}/{args.max_rounds}] "
            f"{args.steps_per_round:,} timesteps",
            flush=True,
        )
        model.learn(
            total_timesteps=args.steps_per_round,
            reset_num_timesteps=False,
            progress_bar=False,
            callback=ProgressCallback(
                args.log_every_steps, args.checkpoint_every_steps
            ),
        )
        model.save(last_model)
        export_sb3_policy(model, last_numpy)
        policy = NumpyLeaguePolicy(last_numpy)
        candidate = evaluate_policy(
            policy.predict,
            args.incumbent,
            opponents,
            seeds=eval_seeds,
            episode_steps=args.episode_steps,
        )
        gate = promotion_gate(baseline, candidate)
        candidate.update(
            {
                "round": len(rounds) + 1,
                "total_timesteps": model.num_timesteps,
                "promotion_gate": gate,
            }
        )
        rounds.append(candidate)
        print(f"[round result] {json.dumps(candidate)}", flush=True)
        ranking = (candidate["score_rate"], candidate["mean_money_margin"])
        if gate["passed"] and ranking > best_ranking:
            shutil.copy2(last_numpy, best_numpy)
            shutil.copy2(last_model, best_model)
            best_candidate = candidate
            best_ranking = ranking
            model.save(anchor_model)
            print("[promotion] Saved a policy stronger than the incumbent", flush=True)
        elif (
            baseline["score_rate"] - candidate["score_rate"]
            >= args.catastrophic_score_drop
        ):
            print(
                "[rollback] Catastrophic evaluation; restoring the last safe "
                "training anchor",
                flush=True,
            )
            model = PPO.load(
                anchor_model, env=vector_environment, device=args.device
            )
            model.save(last_model)

        report = {
            "schema_version": 1,
            "policy_version": LEAGUE_POLICY_VERSION,
            "incumbent": str(args.incumbent),
            "opponent_pool": str(args.opponent_pool),
            "baseline": baseline,
            "curriculum": curriculum,
            "best_candidate": best_candidate,
            "promoted": best_candidate is not None,
            "best_policy": str(best_numpy),
            "rounds": rounds,
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--opponent-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps-per-round", type=int, default=25_000)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--train-envs", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--entropy", type=float, default=0.005)
    parser.add_argument("--incumbent-initial-bias", type=float, default=0.2)
    parser.add_argument("--weakness-bonus", type=float, default=2.0)
    parser.add_argument("--veto-weight-bonus", type=float, default=1.15)
    parser.add_argument("--catastrophic-score-drop", type=float, default=0.25)
    parser.add_argument("--train-seed-offset", type=int, default=100_000)
    parser.add_argument("--eval-seed-offset", type=int, default=9_100_000)
    parser.add_argument("--eval-seed-count", type=int, default=2)
    parser.add_argument("--model-seed", type=int, default=73)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every-steps", type=int, default=2_000)
    parser.add_argument("--checkpoint-every-steps", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
