"""
Watch a trained Snake AI play and record scores.

Usage:
    python test_agent.py                      # uses best_model.zip
    python test_agent.py --model snake_ppo_best
    python test_agent.py --games 20 --fps 10  # slower, fewer games
    python test_agent.py --no-render          # headless benchmark
"""

import argparse
import csv
import sys

import numpy as np
import pygame

from stable_baselines3 import PPO
from snake_env import SnakeEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_model",
                        help="Model file (without .zip)")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--out", default="scores.csv")
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"
    env = SnakeEnv(render_mode=render_mode)
    if render_mode == "human":
        env.metadata["render_fps"] = args.fps

    try:
        model = PPO.load(args.model, env=env)
    except FileNotFoundError:
        print(f"Model '{args.model}.zip' not found. Train first with:")
        print("  python train_agent.py")
        sys.exit(1)

    scores = []
    obs, _ = env.reset()

    for game_num in range(1, args.games + 1):
        done = False
        while not done:
            if render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        pygame.quit()
                        _save_scores(scores, args.out)
                        sys.exit()

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        score = env.game.score
        scores.append(score)
        print(f"Game {game_num:3d}/{args.games}  score={score:3d}  "
              f"avg={np.mean(scores):.1f}")
        obs, _ = env.reset()

    env.close()
    _save_scores(scores, args.out)

    print(f"\n{'─'*40}")
    print(f"Games   : {len(scores)}")
    print(f"Mean    : {np.mean(scores):.2f}")
    print(f"Median  : {float(np.median(scores)):.1f}")
    print(f"Max     : {max(scores)}")
    print(f"≥10 pts : {sum(s >= 10 for s in scores)}/{len(scores)}")
    print(f"Scores saved → {args.out}")


def _save_scores(scores, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game", "score"])
        for i, s in enumerate(scores, 1):
            w.writerow([i, s])


if __name__ == "__main__":
    main()
