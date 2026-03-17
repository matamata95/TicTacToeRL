## Overview
This repository provdes:
- a small RL environment implementation ticktoe_env.py
- a Q-learning agent imeplementation agents/q_learning_agent.py
- simple opponents agents/random_agent.py , agents/minmax_agent.py
- training and play scripts train.py , play.py

Requirements are contained in requirements.yml and environment.yaml, python version = 3.9

## Quick start
- train the agent to generate a q_table.pkl:
    python src/train.py
- play against the trained agent:
    python src/play.py

## Training options
The training fucntion is defined as:
train(episodes=2000, alpha=0.5, gamma=0.99, epsilon=0.1)

- episodes: number of training episodes
- alpha: learning rate
- gamma: discount factor
- epsilon: exploration probability (epsilon-greedy)

To change these parameters edit train.py.

## Playing
play.py loads q_table.pkl and runs a terminal game.
- trained agent plays as player 1 (X)
- human player plays as player 2 (O)
- input moves are integers 0-8 that correspond to the board cells:
    0 1 2
    3 4 5
    6 7 8

## Environment API
- env.reset() - resets the board state and returns observation() - a list containing 9 elements that correspond the state of board cells. 
- env.step(action) -  takes an action and returns next_observation, reward, done,info
- env.available_actions() - list of legal playable moves, if a cell is taken, trying to input its corresponding integer results into an illegal move
- env.render() - prints the board state
- board encoding: 0 = empty, 1 = X (player 1), 2 = O (player 2)

## Q-table
is a dict thats mapping integer state keys to arrays of 9 Q-values. State key encoding is a bse-3 encoding of the board (0/1/2 per cell)