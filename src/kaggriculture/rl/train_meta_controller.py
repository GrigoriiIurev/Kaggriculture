"""Train and evaluate a PPO controller against a fixed expert agent."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .meta_policy import CANDIDATE_NAMES, NumpyMetaPolicy
from .training_env import KaggricultureMetaEnv


def save_fallback_policy(
    path: str | Path,
    feature_count: int,
    hidden_sizes: tuple[int, int] = (128, 64),
) -> None:
    """Write a valid controller that always selects the original expert."""

    first, second = hidden_sizes
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        w0=np.zeros((first, feature_count), dtype=np.float32),
        b0=np.zeros(first, dtype=np.float32),
        w1=np.zeros((second, first), dtype=np.float32),
        b1=np.zeros(second, dtype=np.float32),
        w2=np.zeros((len(CANDIDATE_NAMES), second), dtype=np.float32),
        b2=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        feature_count=np.asarray(feature_count),
        candidate_count=np.asarray(len(CANDIDATE_NAMES)),
    )


def export_sb3_policy(model: Any, path: str | Path) -> None:
    """Export the PPO actor's three linear layers for pure NumPy inference."""

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
        "candidate_count": np.asarray(model.action_space.n),
    }
    for index, layer in enumerate(layers):
        arrays[f"w{index}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{index}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path, **arrays)


def evaluate_selector(
    selector: Callable[[np.ndarray], int],
    expert_path: str | Path,
    games: int,
    episode_steps: int,
    seed_offset: int,
) -> dict[str, Any]:
    """Evaluate paired seats on seeds outside the training range."""

    environment = KaggricultureMetaEnv(
        expert_path,
        episode_steps=episode_steps,
        seed_offset=seed_offset,
        fixed_seats=True,
    )
    wins = ties = losses = 0
    margins: list[float] = []
    candidate_counts = [0] * len(CANDIDATE_NAMES)
    for game in range(games):
        observation, reset_info = environment.reset(seed=game // 2)
        done = False
        info: dict[str, Any] = reset_info
        while not done:
            choice = int(selector(observation))
            candidate_counts[choice] += 1
            observation, _, done, _, info = environment.step(choice)
        outcome = int(info["outcome"])
        wins += outcome > 0
        ties += outcome == 0
        losses += outcome < 0
        margins.append(float(info["money_margin"]))
        print(
            f"[evaluate] {game + 1}/{games} games: "
            f"{wins}W-{ties}T-{losses}L",
            flush=True,
        )
    return {
        "games": games,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / games,
        "score_rate": (wins + 0.5 * ties) / games,
        "mean_money_margin": float(np.mean(margins)),
        "median_money_margin": float(np.median(margins)),
        "candidate_counts": dict(zip(CANDIDATE_NAMES, candidate_counts)),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:  # pragma: no cover - exercised in Colab
        raise ImportError("Install requirements-rl.txt before training") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = KaggricultureMetaEnv(args.expert, episode_steps=args.episode_steps)
    feature_count = probe.observation_space.shape[0]
    best_numpy = args.output_dir / "best_meta_policy.npz"
    last_model = args.output_dir / "last_ppo_model.zip"
    best_model = args.output_dir / "best_ppo_model.zip"
    report_path = args.output_dir / "training_report.json"

    previous_report = None
    if report_path.is_file() and best_numpy.is_file():
        previous_report = json.loads(report_path.read_text(encoding="utf-8"))
        previous_best = previous_report.get("best", {})
        if previous_best.get("win_rate", 0.0) >= args.target_win_rate:
            previous_report["target_win_rate"] = args.target_win_rate
            previous_report["target_met"] = True
            report_path.write_text(
                json.dumps(previous_report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print("[train] Existing best policy already meets the target", flush=True)
            return previous_report
    else:
        save_fallback_policy(best_numpy, feature_count)
    baseline = evaluate_selector(
        lambda _: 0,
        args.expert,
        args.eval_games,
        args.episode_steps,
        args.eval_seed_offset,
    )
    best_metrics = (
        previous_report.get("best", baseline) if previous_report else baseline
    )
    print(f"[baseline] {json.dumps(baseline)}", flush=True)

    factories = []
    for index in range(args.train_envs):
        def factory(index: int = index) -> KaggricultureMetaEnv:
            return KaggricultureMetaEnv(
                args.expert,
                episode_steps=args.episode_steps,
                seed_offset=args.train_seed_offset + index * 1_000_000,
                fixed_seats=True,
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
            gamma=0.999,
            gae_lambda=0.95,
            ent_coef=args.entropy,
            verbose=1,
            device=args.device,
            seed=args.model_seed,
        )

    rounds: list[dict[str, Any]] = (
        list(previous_report.get("rounds", [])) if previous_report else []
    )
    target_met = False
    for round_number in range(1, args.max_rounds + 1):
        print(
            f"[train round {round_number}/{args.max_rounds}] "
            f"{args.steps_per_round:,} timesteps",
            flush=True,
        )
        model.learn(
            total_timesteps=args.steps_per_round,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        model.save(last_model)
        round_numpy = args.output_dir / "last_meta_policy.npz"
        export_sb3_policy(model, round_numpy)
        exported_policy = NumpyMetaPolicy(round_numpy)

        metrics = evaluate_selector(
            exported_policy.predict,
            args.expert,
            args.eval_games,
            args.episode_steps,
            args.eval_seed_offset,
        )
        completed_rounds = len(rounds)
        metrics["round"] = completed_rounds + 1
        metrics["total_timesteps"] = (
            completed_rounds + 1
        ) * args.steps_per_round
        rounds.append(metrics)
        print(f"[round result] {json.dumps(metrics)}", flush=True)
        ranking = (metrics["score_rate"], metrics["mean_money_margin"])
        best_ranking = (
            best_metrics["score_rate"],
            best_metrics["mean_money_margin"],
        )
        if ranking > best_ranking:
            shutil.copy2(round_numpy, best_numpy)
            shutil.copy2(last_model, best_model)
            best_metrics = metrics
            print("[best] Saved a stronger controller", flush=True)
        if metrics["win_rate"] >= args.target_win_rate:
            target_met = True
            print(
                f"[target] Reached {metrics['win_rate']:.1%} win rate",
                flush=True,
            )
            break

    report = {
        "expert": str(args.expert),
        "candidate_names": list(CANDIDATE_NAMES),
        "baseline": baseline,
        "best": best_metrics,
        "target_win_rate": args.target_win_rate,
        "target_met": target_met,
        "rounds": rounds,
        "best_policy": str(best_numpy),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps-per-round", type=int, default=100_000)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--target-win-rate", type=float, default=0.8)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--train-envs", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--train-seed-offset", type=int, default=10_000)
    parser.add_argument("--eval-seed-offset", type=int, default=9_000_000)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
