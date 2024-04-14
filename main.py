import torch
import logging
import tetris_game
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import multiprocessing
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import optim


class Actor:
    pass


class Critic:
    pass


if __name__ == '__main__':
    dtype = torch.float
    # device = torch.device("cpu")
    device = torch.device("cuda:0")