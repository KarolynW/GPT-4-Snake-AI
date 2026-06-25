"""
Gymnasium environment wrapping SnakeGame.

Observation: 11-dimensional binary float32 vector
Actions:     Discrete(3) — 0=straight, 1=turn-right, 2=turn-left
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from snake_game import SnakeGame, GRID_W, GRID_H

CELL = 28          # pixels per grid cell
WIN_W = GRID_W * CELL
WIN_H = GRID_H * CELL + 40  # extra bar for score

_COLOR_BG    = (15,  15,  15)
_COLOR_GRID  = (30,  30,  30)
_COLOR_FOOD  = (220, 50,  50)
_COLOR_HEAD  = (50,  200, 80)
_COLOR_BODY  = (30,  140, 60)
_COLOR_TEXT  = (220, 220, 220)


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(0.0, 1.0, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.game = SnakeGame()
        self._screen = None
        self._clock = None
        self._font = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game = SnakeGame(seed=seed)
        else:
            self.game.reset()
        obs = self.game.get_state()
        if self.render_mode == "human":
            self._render_frame()
        return obs, {}

    def step(self, action: int):
        reward, terminated = self.game.step(int(action))
        truncated = False
        obs = self.game.get_state()
        info = {"score": self.game.score}
        if self.render_mode == "human":
            self._render_frame()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        self._render_frame()

    def close(self):
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_frame(self):
        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((WIN_W, WIN_H))
                pygame.display.set_caption("Snake AI")
            else:
                self._screen = pygame.Surface((WIN_W, WIN_H))
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("monospace", 22, bold=True)

        surf = self._screen
        surf.fill(_COLOR_BG)

        # Subtle grid lines
        for x in range(0, WIN_W, CELL):
            pygame.draw.line(surf, _COLOR_GRID, (x, 0), (x, GRID_H * CELL))
        for y in range(0, GRID_H * CELL, CELL):
            pygame.draw.line(surf, _COLOR_GRID, (0, y), (WIN_W, y))

        # Food
        fx, fy = self.game.food
        if fx >= 0:
            pygame.draw.ellipse(
                surf, _COLOR_FOOD,
                (fx * CELL + 3, fy * CELL + 3, CELL - 6, CELL - 6)
            )

        # Snake body (draw tail → head so head is on top)
        for i, (sx, sy) in enumerate(reversed(self.game.snake)):
            color = _COLOR_HEAD if i == len(self.game.snake) - 1 else _COLOR_BODY
            pygame.draw.rect(surf, color, (sx * CELL + 1, sy * CELL + 1, CELL - 2, CELL - 2), border_radius=4)

        # Score bar
        score_surf = self._font.render(
            f"Score: {self.game.score}   Steps: {self.game.steps}", True, _COLOR_TEXT
        )
        surf.blit(score_surf, (8, GRID_H * CELL + 8))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
            )
