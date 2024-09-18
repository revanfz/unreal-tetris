import gym_tetris

import torch
import torch.nn as nn
import torch.nn.functional as F

from time import sleep
from model import UNREAL
from optimizer import SharedRMSprop
from wrapper import FrameSkipWrapper
from replay_buffer import ReplayBuffer
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import MOVEMENT
from torch.distributions import Categorical
from utils import make_env, preprocessing, ensure_share_grads

params = {
    "lr": 0.0005,
    "unroll_steps": 20,
    "beta": 0.00067,
    "gamma": 0.99,
    "hidden_size": 256, 
    "task_weight": 0.01
}

device = torch.device("cpu")

if __name__ == "__main__":
    env = make_env(resize=None, grayscale=False, framestack=None, render_mode="human")

    global_model = UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=env.action_space.n,
        hidden_size=256,
        device=device,
    )
    optimizer = SharedRMSprop(global_model.parameters(), params["lr"])
    local_model = UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=env.action_space.n,
        hidden_size=256,
        device=device,
    )
    experience_replay = ReplayBuffer(2000)

    done = True
    num_games = 0
    block_placed = 0
    last_lines = 0

    while True:
        optimizer.zero_grad()
        local_model.load_state_dict(global_model.state_dict())

        for step in range(params["unroll_steps"]):
            if done:
                state, info = env.reset()
                last_lines = 0
                state = preprocessing(state)
                prev_action = torch.zeros(1, env.action_space.n).to(device)
                prev_reward = torch.zeros(1, 1).to(device)
                hx = torch.zeros(1, params["hidden_size"]).to(device)
                cx = torch.zeros(1, params["hidden_size"]).to(device)

                if num_games:
                    block_placed += episode_blocks
                    print(f"Block placed: {block_placed / num_games}, rewards: {episode_reward}")

                num_games += 1
                episode_reward = 0
                episode_blocks = 0
                
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            policy, _, _, hx, cx = local_model(
                state_tensor, prev_action, prev_reward, (hx, cx)
            )

            dist = Categorical(policy)
            action = dist.sample().detach()

            next_state, reward, done, _, info = env.step(9)
            print(info)
            episode_blocks = sum(info["statistics"].values())
            next_state = preprocessing(next_state)
            episode_reward += reward

            experience_replay.store(
                state,
                prev_action.argmax().item(),
                prev_reward.item(),
                next_state,
                action.item(),
                reward,
                done,
            )

            prev_action = F.one_hot(action, num_classes=env.action_space.n).to(
                device
            )
            prev_reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
            state = next_state

        # Hitung loss A3C
        # 1. Sampel replay buffer secara sekuensial
        states, prev_actions, prev_rewards, _, _, _, dones = experience_replay.sample(
            params["unroll_steps"], base=True
        )
        # 2. Hitung loss actor dan critic
        policy_loss, value_loss, entropy = local_model.a3c_loss(
            states, prev_rewards, prev_actions, dones
        )
        # 3. Jumlahkan loss dengan mengurangi nilai critic loss
        a3c_loss = policy_loss + value_loss

        # Hitung Loss Pixel Control dan Feature Control
        # 1.  Sampling replay buffer secara random
        states, prev_actions, prev_rewards, next_states, next_actions, next_rewards, dones = experience_replay.sample(
            params["unroll_steps"]
        )
        # 2a. Hitung loss Pixel Control
        aux_control_loss = local_model.control_loss(
            states, prev_actions, prev_rewards, next_states, next_actions, next_rewards, dones
        )  
        # 2b. Hitung loss Value Replay
        vr_loss = local_model.vr_loss(states, prev_actions, prev_rewards, dones)

        # Hitung Loss Reward Pedictions
        # 1. Sampel 3 frame dengan pleuang rewarding state = 0.5
        states, prev_rewards, next_rewards = experience_replay.sample_rp()
        # 2. Hitung loss reward prediction
        rp_loss = local_model.rp_loss(states, prev_rewards, next_rewards[-1])

        # Penjumlahan loss a3c, pixel control, value replay dan reward prediction
        total_loss = (
            a3c_loss + vr_loss + params["task_weight"] * aux_control_loss + rp_loss
        )

        total_loss.backward()
        nn.utils.clip_grad_norm_(local_model.parameters(), 40)
        ensure_share_grads(
            local_model=local_model, global_model=global_model, device=device
        )
        optimizer.step()

        # break
