# GPT-4-Snake-AI
This repository contains a set of scripts created by GPT-4, an advanced AI language model by OpenAI. The project demonstrates how AI can create a simple AI to play the classic Snake game.

## Overview
The project uses the following key components:

* snake_game.py: This script defines the basic Snake game logic, including the Snake and Food classes, as well as the game's grid dimensions and colors.
* snake_env.py: This script defines a custom Gym environment for the Snake game, making it compatible with reinforcement learning algorithms.
* train_agent.py: This script trains a Snake-playing agent using the PPO reinforcement learning algorithm from the Stable Baselines3 library.
* test_agent.py: This script allows you to watch the trained agent play the Snake game and records its performance in a CSV file.

##Installation

Clone the repository:

git clone https://github.com/your-username/GPT-4-Snake-AI.git

Change to the project directory:

cd GPT-4-Snake-AI

Create a Python virtual environment (optional but recommended):

python3 -m venv venv

Activate the virtual environment:

On Windows: venv\Scripts\activate

On macOS/Linux: source venv/bin/activate
Install the required packages:

pip install -r requirements.txt

## Usage

Train the agent
To train the agent, run the train_agent.py script:

python train_agent.py

This script will train the agent for a specified number of timesteps and periodically evaluate its performance, rendering the game and updating a plot of the mean reward.

After training, the script will save the trained agent as snake_ppo.zip.

## Watch the agent play

To watch the trained agent play the Snake game, run the test_agent.py script:

python test_agent.py

The script will load the trained agent from snake_ppo.zip and play the game, showing the agent's performance in real-time.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
