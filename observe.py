from pprint import pp
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model import UNREAL
from optimizer import SharedRMSprop
from replay_buffer import ReplayBuffer
from torch.distributions import Categorical
from utils import make_env, pixel_diff, preprocessing, ensure_share_grads

# params = dict(
#     lr=0.0005,
#     unroll_steps=20,
#     beta=0.00067,
#     gamma=0.99,
#     hidden_size=256,
#     task_weight=0.01,
# )
params = dict(
    lr=0.0003,
    unroll_steps=20,
    beta=0.01,
    gamma=0.99,
    hidden_size=256,
    task_weight=1.0,
)

device = torch.device("cpu")

if __name__ == "__main__":
    env = make_env(resize=84, render_mode="human", level=19, skip=2)
    checkpoint = torch.load("trained_models/final.pt", weights_only=True)

    local_model = UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=env.action_space.n,
        hidden_size=256,
        device=device,
    )
    local_model.load_state_dict(
        checkpoint
    )
    local_model.eval()

    done = True
    action = F.one_hot(torch.LongTensor([0]), num_classes=env.action_space.n).to(device)
    reward = torch.zeros(1, 1).to(device)


    while True:
        if done:
            state, info = env.reset()
            state = preprocessing(state)
            hx = torch.zeros(1, params["hidden_size"]).to(device)
            cx = torch.zeros(1, params["hidden_size"]).to(device)
            eps_r = []
        else:
            hx = hx.detach()
            cx = cx.detach()

        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        policy, value, hx, cx = local_model(state_tensor, action, reward, (hx, cx))

        dist = Categorical(probs=policy)
        action = dist.sample()

        next_state, reward, done, _, info = env.step(action.item())
        state = preprocessing(next_state)
        action = F.one_hot(action, num_classes=env.action_space.n).to(device)
        reward = torch.FloatTensor([[reward]]).to(device)

        if done:
            print(info)
            break