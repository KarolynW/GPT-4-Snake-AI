"""
Train a Snake AI using PPO (Proximal Policy Optimization).

Key design choices:
- 11-feature compact state: danger×3, direction×4, food×4
- Relative 3-action space (straight/right/left) — no invalid reversal possible
- 8 parallel environments for throughput
- Distance shaping + timeout penalty for robust reward signal
- 2 million timesteps — sufficient for strong play on a 20×20 grid

Usage:
    python train_agent.py
    python train_agent.py --timesteps 5000000  # longer for peak performance
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from snake_env import SnakeEnv


def make_env():
    return Monitor(SnakeEnv())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                        help="Total training timesteps (default 2M)")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--out", type=str, default="snake_ppo_best",
                        help="Output model name (no extension)")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Vectorised training environments
    train_env = make_vec_env(make_env, n_envs=args.n_envs)

    # Separate eval environment (single, deterministic)
    eval_env = Monitor(SnakeEnv())

    # Save best model whenever eval improves
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=".",
        log_path="logs",
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    # Periodic checkpoint so training can be resumed
    ckpt_cb = CheckpointCallback(
        save_freq=max(200_000 // args.n_envs, 1),
        save_path="checkpoints",
        name_prefix="snake_ppo",
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log="logs/tb",
    )

    print(f"Training for {args.timesteps:,} timesteps across {args.n_envs} envs…")
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([eval_cb, ckpt_cb]),
        progress_bar=True,
    )

    model.save(args.out)
    print(f"\nTraining complete. Best model saved to best_model.zip")
    print(f"Final model saved to {args.out}.zip")
    print(f"Run `python test_agent.py` to watch it play.")

    # Quick final evaluation
    print("\nFinal evaluation (50 episodes)…")
    scores = []
    obs, _ = eval_env.reset()
    ep_score = 0
    for _ in range(50 * 1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        if "score" in info:
            ep_score = info["score"]
        if terminated or truncated:
            scores.append(ep_score)
            ep_score = 0
            obs, _ = eval_env.reset()
            if len(scores) >= 50:
                break

    if scores:
        print(f"Mean score: {np.mean(scores):.1f}  "
              f"Max: {max(scores)}  "
              f"≥10: {sum(s >= 10 for s in scores)}/{len(scores)}")


if __name__ == "__main__":
    main()
