#!/usr/bin/env python3
"""Load a previously trained PPO model and watch it control Pendulum-v1."""

import argparse
import gymnasium as gym
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="ppo_clip_output/ppo_pendulum.zip",
        help="Path to saved Stable-Baselines3 PPO model",
    )
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = gym.make(args.env_id, render_mode="human")

    try:
        for episode in range(1, args.episodes + 1):
            obs, info = env.reset()
            terminated = truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)

            print(f"Episode {episode}: return = {total_reward:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
