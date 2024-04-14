import random
import numpy as np
import tetris_game
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd

import tetris_game
import tetris_game.tetris.tetris


def simulate():
    global epsilon, epsilon_decay
    for episode in range(MAX_EPISODES):
        state, info = env.reset()
        total_reward = 0

        for t in range(MAX_TRY):

            action = env.action_space.sample()

            n_state, reward, done, _, info = env.step(action)
            env.render()

            if done or t >= MAX_TRY - 1:
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, reward))
                break

        # exploring rate decay
        # if epsilon >= 0.005:
        #     epsilon *= epsilon_decay


if __name__ == "__main__":
    env = gym.make("SmartTetris-v0")
    MAX_EPISODES = 999
    MAX_TRY = 1000
    # epsilon = 1
    # epsilon_decay = 0.999
    # learning_rate = 0.1
    # gamma = 0.6
    # num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    # q_table = np.zeros(num_box + (env.action_space.n,))
    # simulate()
    tetris = tetris_game.tetris.tetris.Tetris()
    tetris.run()
    
