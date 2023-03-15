import csv
import pygame
import gym
from stable_baselines3 import PPO
from snake_env import SnakeEnv, GRID_SIZE, GRID_WIDTH, GRID_HEIGHT, WHITE, GREEN, RED
import sys
import numpy as np

# Initialize Pygame
pygame.init()

# Display settings
WIDTH, HEIGHT = GRID_WIDTH * GRID_SIZE, GRID_HEIGHT * GRID_SIZE

# Pygame setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

def draw_board(env):
    screen.fill(WHITE)

    snake = env.snake
    food = env.food

    for x, y in snake.body:
        pygame.draw.rect(screen, GREEN, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    food_x, food_y = food.position
    pygame.draw.rect(screen, RED, (food_x * GRID_SIZE, food_y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    pygame.display.flip()

# Create the environment
env = SnakeEnv()

# Load the trained model
model = PPO.load("snake_ppo")

num_games = 100
scores = []

for _ in range(num_games):
    obs = env.reset()
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

        env.render()
        draw_board(env)
        print(f"Current score: {env.score}")

        clock.tick(1000)

    scores.append(env.score)

# Calculate the average score
average_score = np.mean(scores)

# Save the scores to a CSV file
with open("scores.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Game", "Score"])
    for i, score in enumerate(scores):
        csvwriter.writerow([i + 1, score])

print(f"Average score over {num_games} games: {average_score}")


