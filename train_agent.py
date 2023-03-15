import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from snake_env import SnakeEnv
import numpy as np
import pygame
import matplotlib.pyplot as plt

print("Starting train_agent.py")

mean_rewards_history = []

# Turn on interactive mode
plt.ion()

# Create an initial plot
fig, ax = plt.subplots()
ax.set_xlabel('Evaluation point')
ax.set_ylabel('Mean reward')
ax.set_title('Mean reward at each evaluation point')

def evaluate_and_render(model, env, n_episodes=10):
    mean_rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            env.render()
            episode_reward += reward
        mean_rewards.append(episode_reward)
    mean_reward = np.mean(mean_rewards)
    std_reward = np.std(mean_rewards)
    return mean_reward, std_reward

# Create the training environment
env = DummyVecEnv([lambda: SnakeEnv(render=False)])

# Create the evaluation environment with a 100ms delay during rendering
eval_env = SnakeEnv(render=True)

# Hyperparameters
learning_rate = 0.0005
n_steps = 256
gae_lambda = 0.95
gamma = 0.99
ent_coef = 0.01
clip_range = 0.2
n_epochs = 10
batch_size = 64

# Create the PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=learning_rate,
    n_steps=n_steps,
    gae_lambda=gae_lambda,
    gamma=gamma,
    ent_coef=ent_coef,
    clip_range=clip_range,
    n_epochs=n_epochs,
    batch_size=batch_size,
)

# Train the model
total_timesteps = 100000
log_interval = 10000
for timestep in range(0, total_timesteps, log_interval):
    print(f"Training for {log_interval} timesteps")
    model.learn(total_timesteps=log_interval)
    print("Evaluating model")
    mean_reward, std_reward = evaluate_and_render(model, eval_env, n_episodes=10)
    print(f"Timestep: {timestep}, Mean reward: {mean_reward} +/- {std_reward}")

    # Update the plot
    mean_rewards_history.append(mean_reward)
    ax.clear()
    ax.plot(mean_rewards_history)
    ax.set_xlabel('Evaluation point')
    ax.set_ylabel('Mean reward')
    ax.set_title('Mean reward at each evaluation point')
    plt.pause(0.01)  # Add a small delay to allow the plot to update

# Save the model
model.save("snake_ppo")

# Turn off interactive mode
plt.ioff()

# Evaluate the trained agent
print("Evaluating final model")
mean_reward, std_reward = evaluate_and_render(model, eval_env, n_episodes=10)
print(f"Final Mean reward: {mean_reward} +/- {std_reward}")

