"""
Core Snake game logic — no rendering, no gym dependencies.

Grid coordinates: (col, row) with (0,0) at top-left.
Direction integers: 0=RIGHT 1=DOWN 2=LEFT 3=UP
Action integers:   0=straight 1=turn-right 2=turn-left (relative to current direction)
"""

import numpy as np

GRID_W = 20
GRID_H = 20

# (dx, dy) for directions 0..3
_DX = [1, 0, -1, 0]
_DY = [0, 1, 0, -1]


class SnakeGame:
    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        cx, cy = GRID_W // 2, GRID_H // 2
        self.direction = 0  # start going RIGHT
        self.snake: list[tuple[int, int]] = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self._snake_set: set[tuple[int, int]] = set(self.snake)
        self._place_food()

    def step(self, action: int) -> tuple[float, bool]:
        """
        Apply a relative action (0=straight, 1=right, 2=left).
        Returns (reward, done).
        """
        if action == 1:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            self.direction = (self.direction - 1) % 4

        self.steps += 1
        self.steps_since_food += 1

        head = self.snake[0]
        new_head = (head[0] + _DX[self.direction], head[1] + _DY[self.direction])

        # Wall collision
        if not (0 <= new_head[0] < GRID_W and 0 <= new_head[1] < GRID_H):
            return -10.0, True

        # Self collision — exclude tail since it will vacate this step
        if new_head in self._snake_set and new_head != self.snake[-1]:
            return -10.0, True

        # Move snake
        tail = self.snake.pop()
        self._snake_set.discard(tail)
        self.snake.insert(0, new_head)
        self._snake_set.add(new_head)

        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            # Regrow by re-appending the tail
            self.snake.append(tail)
            self._snake_set.add(tail)
            self._place_food()
            return 10.0, False

        # Timeout: prevent the snake from looping forever
        timeout = GRID_W * GRID_H * (1 + self.score)
        if self.steps_since_food >= timeout:
            return -10.0, True

        # Small distance-based shaping
        prev_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        curr_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        return (0.1 if curr_dist < prev_dist else -0.1), False

    def get_state(self) -> np.ndarray:
        """Return the 11-feature binary state vector."""
        head = self.snake[0]
        d = self.direction

        # Relative danger checks
        def _danger(dd: int) -> float:
            nx = head[0] + _DX[dd]
            ny = head[1] + _DY[dd]
            if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                return 1.0
            # Exclude tail for the same reason as in step()
            body_no_tail = self._snake_set - {self.snake[-1]}
            return 1.0 if (nx, ny) in body_no_tail else 0.0

        danger_s = _danger(d)
        danger_r = _danger((d + 1) % 4)
        danger_l = _danger((d - 1) % 4)

        fx, fy = self.food
        hx, hy = head
        return np.array([
            danger_s, danger_r, danger_l,
            float(d == 0), float(d == 1), float(d == 2), float(d == 3),  # direction one-hot
            float(fx < hx), float(fx > hx),  # food left / right
            float(fy < hy), float(fy > hy),  # food up  / down
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _place_food(self):
        candidates = [
            (x, y)
            for x in range(GRID_W)
            for y in range(GRID_H)
            if (x, y) not in self._snake_set
        ]
        if not candidates:
            # Board full — snake wins; treat as done at env level
            self.food = (-1, -1)
            return
        idx = int(self._rng.integers(len(candidates)))
        self.food = candidates[idx]
