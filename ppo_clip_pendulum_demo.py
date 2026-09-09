#!/usr/bin/env python3
"""
PPO-Clip continuous-control classroom demo.

Trains PPO-Clip on Gymnasium Pendulum-v1 with hyperparameters chosen to mirror
the classic PPO continuous-control configuration where practical.

Metrics:
- mean episodic return +/- std over evaluation episodes
- periodic evaluation curve saved to PNG/NPZ
- TensorBoard training logs

F1 is intentionally not used: this is reinforcement learning, not classification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def random_policy_return(env_id: str, episodes: int, seed: int) -> tuple[float, float]:
    returns = []

    for episode in range(episodes):
        env = gym.make(env_id)
        obs, info = env.reset(seed=seed + episode)
        env.action_space.seed(seed + episode)

        terminated = truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

        returns.append(total_reward)
        env.close()

    return float(np.mean(returns)), float(np.std(returns))


def make_eval_env(env_id: str) -> Monitor:
    return Monitor(gym.make(env_id))


def plot_evaluations(npz_path: Path, output_path: Path) -> None:
    if not npz_path.exists():
        print(f"Evaluation file not found: {npz_path}")
        return

    data = np.load(npz_path)
    timesteps = data["timesteps"]
    results = data["results"]
    means = results.mean(axis=1)
    stds = results.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, means, marker="o")
    plt.fill_between(timesteps, means - stds, means + stds, alpha=0.2)
    plt.xlabel("Environment timesteps")
    plt.ylabel("Mean episodic return")
    plt.title("PPO-Clip evaluation during training")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def render_trained_policy(model: PPO, env_id: str, episodes: int = 3) -> None:
    env = gym.make(env_id, render_mode="human")

    try:
        for episode in range(episodes):
            obs, info = env.reset()
            terminated = truncated = False

            while not (terminated or truncated):
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    out_dir = Path("ppo_clip_output")
    out_dir.mkdir(exist_ok=True)

    print("\n=== RANDOM POLICY BASELINE ===")
    random_mean, random_std = random_policy_return(
        args.env_id,
        args.eval_episodes,
        args.seed,
    )
    print(f"Random policy return: {random_mean:.2f} +/- {random_std:.2f}")

    train_env = Monitor(gym.make(args.env_id))
    eval_env = make_eval_env(args.env_id)

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(out_dir / "tensorboard"),
        verbose=1,
        seed=args.seed,
        device="cpu",
    )

    print("\n=== UNTRAINED PPO POLICY ===")
    pre_mean, pre_std = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )
    print(f"Untrained PPO return: {pre_mean:.2f} +/- {pre_std:.2f}")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    print("\n=== TRAINING ===")
    print("Algorithm: PPO-Clip")
    print("Learning rate: 3e-4")
    print("Rollout horizon: 2048")
    print("Minibatch size: 64")
    print("Update epochs per rollout: 10")
    print("Gamma: 0.99")
    print("GAE lambda: 0.95")
    print("Clip epsilon: 0.2")
    print(f"Total timesteps: {args.timesteps}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
    )

    model_path = out_dir / "ppo_pendulum"
    model.save(model_path)

    print("\n=== FINAL EVALUATION ===")
    final_mean, final_std = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    print(f"Final PPO return: {final_mean:.2f} +/- {final_std:.2f}")
    print(
        "Improvement over untrained PPO: "
        f"{final_mean - pre_mean:+.2f} return points"
    )
    print(
        "Improvement over random baseline: "
        f"{final_mean - random_mean:+.2f} return points"
    )

    eval_npz = out_dir / "eval" / "evaluations.npz"
    curve_path = out_dir / "ppo_learning_curve.png"
    plot_evaluations(eval_npz, curve_path)

    print(f"\nSaved model: {model_path}.zip")
    print(f"Learning curve: {curve_path}")
    print(f"TensorBoard logs: {out_dir / 'tensorboard'}")
    print(
        "\nFor Pendulum-v1, rewards are non-positive and 0 is ideal, "
        "so less-negative returns are better."
    )

    train_env.close()
    eval_env.close()

    if args.render:
        render_trained_policy(model, args.env_id)


if __name__ == "__main__":
    main()
