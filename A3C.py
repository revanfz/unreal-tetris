from __future__ import annotations

import os
import torch
import custom_env
import numpy as np
import gymnasium as gym
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from agent import Worker
from optimizer import SharedAdam
from actor_critic import ActorCritic
from torch.multiprocessing import Queue

save_weights = False
load_weights = False

actor_weights_path = "models/actor_weights.h5"
critic_weights_path = "models/critic_weights.h5"


if __name__ == "__main__":
    lr = 1e-4
    env_id = "SmartTetris-v0"
    env = gym.make(env_id)
    n_actions = env.action_space.n
    n_agents = 8
    del env

    global_actor_critic = ActorCritic(
        # input_dims, 
        n_actions
    )
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, betas=(0.92, 0.999))

    if load_weights:
        global_actor_critic.policy.load_state_dict(torch.load(actor_weights_path))
        global_actor_critic.value.load_state_dict(torch.load(critic_weights_path))
        global_actor_critic.policy.eval()
        global_actor_critic.value.eval()
    else:
        global_actor_critic.share_memory()

    all_rewards = []
 
    global_ep = mp.Value("i", 0)
    res_queue = Queue()

    workers = [
        Worker(
            global_actor_critic,
            optim,
            gamma=0.99,
            beta=0.01,
            lr=lr,
            name=i,
            global_eps_idx=global_ep,
            env_id=env_id,
        )
        for i in range(n_agents)
    ]

    [w.start() for w in workers]
    [w.join() for w in workers]

    if not os.path.exists("models"):
        os.makedirs("models")

    if save_weights:
        torch.save(global_actor_critic.policy.state_dict(), actor_weights_path)
        torch.save(global_actor_critic.value.state_dict(), critic_weights_path)
