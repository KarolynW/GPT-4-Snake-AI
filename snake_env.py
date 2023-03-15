import gym
from gym import spaces
import numpy as np
import pygame
from snake_game import Snake, Food, GRID_WIDTH, GRID_HEIGHT, WIDTH, HEIGHT, GRID_SIZE, WHITE, GREEN, RED, BLACK, draw_text


class SnakeEnv(gym.Env):
    def __init__(self, render=False, delay=0):
        super(SnakeEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(GRID_HEIGHT, GRID_WIDTH, 4), dtype=np.float32)
        self.snake = Snake()
        self.food = Food(self.snake.body)
        self.score = 0
        self.delay = delay

        self.render_mode = render
        if render:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Snake AI")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

    def step(self, action):
        direction = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        if direction != (-self.snake.direction[0], -self.snake.direction[1]):
            self.snake.change_direction(direction)

        self.snake.move()

        reward = 0
        done = False

        if self.snake.collides_with_bounds() or self.snake.collides_with_itself():
            done = True
            reward = -1000  # Increase the negative reward for dying
        elif self.snake.body[0] == self.food.position:
            self.snake.grow()
            self.food = Food(self.snake.body)
            self.score += 1
            reward = 500  # Increase the reward for eating food
        else:
            reward = -1  # Increase the negative reward for every step without eating food


        observation = self._get_observation()

        if self.render_mode:
            self.render()

        return observation, reward, done, {}


    def reset(self):
        self.snake = Snake()
        self.food = Food(self.snake.body)
        self.score = 0
        return self._get_observation()

    def render(self, mode='human'):
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((GRID_WIDTH * GRID_SIZE, GRID_HEIGHT * GRID_SIZE))
            pygame.display.set_caption("Snake Agent")
            self.clock = pygame.time.Clock()

        self.screen.fill(WHITE)

        for x, y in self.snake.body:
            pygame.draw.rect(self.screen, GREEN, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        food_x, food_y = self.food.position
        pygame.draw.rect(self.screen, RED, (food_x * GRID_SIZE, food_y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        pygame.display.flip()
        self.clock.tick(500)
        pygame.time.delay(self.delay)

    def _get_observation(self):
        observation = np.zeros((GRID_HEIGHT, GRID_WIDTH, 4), dtype=np.float32)
        for y, row in enumerate(observation):
            for x, _ in enumerate(row):
                if (x, y) in self.snake.body:
                    observation[y, x] = [1, 0, 0, 0]
                elif (x, y) == self.food.position:
                    observation[y, x] = [0, 1, 0, 0]

        head_x, head_y = self.snake.body[0]
        food_x, food_y = self.food.position

        # Add the relative position of the food to the observation
        food_relative_x = food_x - head_x
        food_relative_y = food_y - head_y

        # Normalize the relative position
        food_relative_x /= GRID_WIDTH
        food_relative_y /= GRID_HEIGHT

        # Add the relative position of the food to the observation (can replace channels 2 and 3)
        observation[:, :, 2] = food_relative_x
        observation[:, :, 3] = food_relative_y

        return observation




# Register the Snake environment
gym.envs.registration.register(
    id='Snake-v0',
    entry_point='snake_env:SnakeEnv',
)

