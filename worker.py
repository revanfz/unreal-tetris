import gym_tetris
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import UNREAL
from argparse import Namespace
from optimizer import SharedAdam
from replay_buffer import ReplayBuffer
from wrapper import ActionRepeatWrapper
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
from utils import ensure_share_grads, preprocess_frame_stack
from torch.distributions import Categorical
from multiprocessing.sharedctypes import Synchronized


def worker(
    rank: int,
    global_model: UNREAL,
    optimizer: SharedAdam,
    shared_replay_buffer: ReplayBuffer,
    global_steps: Synchronized,
    params: Namespace,
):
    torch.manual_seed(42 + rank)

    env = gym_tetris.make(
        "TetrisA-v3",
        apply_api_compatibility=True,
        render_mode="human" if not rank else "rgb_array",
    )
    env = JoypadSpace(env, MOVEMENT)
    env = ActionRepeatWrapper(env)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model = UNREAL(
        n_inputs=(3, 84, 84),
        n_actions=env.action_space.n,
        hidden_size=256,
        device=device,
    )
    local_model.train()
    local_replay_buffer = ReplayBuffer(params["unroll_steps"])

    done = True

    while global_steps.value <= params["max_steps"]:
        optimizer.zero_grad()
        local_model.load_state_dict(global_model.state_dict())

        if done:
            state, info = env.reset()
            local_replay_buffer.clear()
            prev_action = torch.zeros(1, env.action_space.n)
            prev_reward = torch.zeros(1, 1)
            hx = torch.zeros(1, params["hidden_size"])
            cx = torch.zeros(1, params["hidden_size"])
            episode_reward = 0

        for _ in range(params["unroll_steps"]):
            if not rank:
                env.render()
            state = preprocess_frame_stack(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            policy, _, _, _, (hx, cx) = local_model(
                state_tensor, prev_action, prev_reward, (hx, cx)
            )

            dist = Categorical(policy)
            action = dist.sample().detach()

            next_state, reward, done, _, info = env.step(action.item())
            next_state = preprocess_frame_stack(next_state)

            prev_action = F.one_hot(action, num_classes=env.action_space.n)
            prev_reward = torch.FloatTensor([reward]).unsqueeze(0)
            shared_replay_buffer.push(
                state, action.item(), reward, next_state, done, True
            )

            episode_reward += reward
            state = next_state

            if done:
                break

        # Hitung loss A3C
        # 1. Hitung returns
        state, action, reward, next_state, done = local_replay_buffer.sample(20)
        actions_oh = F.one_hot(action, num_classes=env.action_space.n)
        a3c_loss = local_model.a3c_loss(state, actions_oh, reward, done, action)

        a3c_loss.backward()
        nn.utils.clip_grad_norm_(local_model.parameters(), 40)
        ensure_share_grads(
            local_model=local_model, global_model=global_model, device=device
        )
        optimizer.step()

        # Hitung Loss Pixel Control dan Feature Control

        # Hitung Loss Reward Pedictions

        # Propagasi Balik total loss
