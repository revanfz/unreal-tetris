import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from model import UNREAL
from optimizer import SharedRMSprop
from replay_buffer import ReplayBuffer
from torch.distributions import Categorical
from utils import make_env, pixel_diff, preprocessing, ensure_share_grads

params = dict(
    lr=0.00036,
    unroll_steps=20,
    beta=0.0113,
    gamma=0.99,
    hidden_size=256,
    pc_weight=0.02335,
)

device = torch.device("cpu")

if __name__ == "__main__":
    env = make_env(
        resize=84,
        render_mode="human",
        level=19,
        skip=2,
        id="TetrisA-v3",
    )

    checkpoint = torch.load(f"./trained_models/UNREAL-tetris_checkpoint.tar", weights_only=True)

    global_model = UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=env.action_space.n,
        hidden_size=params["hidden_size"],
        device=torch.device("cpu"),
        beta=params["beta"],
        gamma=params["gamma"],
        temperature=10.0
    )
    global_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = SharedRMSprop(global_model.parameters(), params["lr"])
    local_model = UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=env.action_space.n,
        hidden_size=256,
        device=device,
        temperature=150.0
    )
    local_model.train()
    experience_replay = ReplayBuffer(500)

    state, info = env.reset()
    state = preprocessing(state)
    prev_action = F.one_hot(torch.LongTensor([0]), env.action_space.n).to(device)
    prev_reward = torch.zeros(1, 1).to(device)
    hx = torch.zeros(1, params["hidden_size"]).to(device)
    cx = torch.zeros(1, params["hidden_size"]).to(device)

    # while not experience_replay._is_full():
    for i in range(100):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            policy, value, hx, cx = local_model(
                state_tensor, prev_action, prev_reward, (hx, cx)
            )

            dist = Categorical(probs=policy)
            action = dist.sample().cpu()

        next_state, reward, done, _, info = env.step(action.item())
        next_state = preprocessing(next_state)
        pixel_change = pixel_diff(state, next_state)
        experience_replay.store(state, reward, action.item(), done, pixel_change)
        state = next_state
        prev_action = F.one_hot(action, num_classes=env.action_space.n).to(device)
        prev_reward = torch.FloatTensor([[reward]]).to(device)

        hx = hx.detach()
        cx = cx.detach()

        if done:
            state, info = env.reset()
            state = preprocessing(state)
            hx = torch.zeros(1, params["hidden_size"]).to(device)
            cx = torch.zeros(1, params["hidden_size"]).to(device)

    done = True

    action = F.one_hot(torch.LongTensor([0]), env.action_space.n).to(device)
    reward = torch.zeros(1, 1).to(device)

    rewards = []
    eps_r = 0

    # for step in tqdm(range(50000), desc="Testing"):
    for step in range(5000):
        optimizer.zero_grad()
        local_model.load_state_dict(global_model.state_dict())

        dones = torch.zeros(params["unroll_steps"], device=device)
        rewards = torch.zeros_like(dones, device=device)
        log_probs = torch.zeros_like(dones, device=device)
        entropies = torch.zeros_like(dones, device=device)
        values = torch.zeros_like(dones, device=device)

        for step in range(params["unroll_steps"]):
            if done:
                state, info = env.reset(seed=42)
                state = preprocessing(state)
                hx = torch.zeros(1, params["hidden_size"], device=device)
                cx = torch.zeros(1, params["hidden_size"], device=device)
                eps_r = 0
            else:
                hx = hx.detach()
                cx = cx.detach()

            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            policy, value, hx, cx = local_model(state_tensor, action, reward, (hx, cx))
            print(policy)

            dist = Categorical(probs=policy)
            action = dist.sample()
            entropy = dist.entropy()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _, info = env.step(action.cpu().item())
            eps_r += reward
            next_state = preprocessing(next_state)
            pixel_change = pixel_diff(state, next_state)
            experience_replay.store(
                state, reward, action.cpu().item(), done, pixel_change
            )

            values[step] = torch.squeeze(value)
            entropies[step] = entropy
            log_probs[step] = torch.squeeze(log_prob)
            dones[step] = torch.tensor(not done, device=device)
            rewards[step] = torch.tensor(reward, device=device)

            state = next_state
            action = F.one_hot(action, num_classes=env.action_space.n).to(device)
            reward = torch.FloatTensor([[reward]]).to(device)

        # Bootstrapping
        R = 0.0
        with torch.no_grad():
            _, R, _, _ = local_model(
                torch.FloatTensor(next_state).unsqueeze(0).to(device),
                action,
                reward,
                (hx, cx),
            )

        # Hitung loss A3C
        actor_loss, critic_loss = local_model.a3c_loss(
            rewards=rewards,
            R=R,
            dones=dones,
            log_probs=log_probs,
            entropies=entropies,
            values=values,
        )
        a3c_loss = actor_loss + 0.5 * critic_loss
        episode_rewards = rewards.sum().detach().cpu().numpy()

        # Hitung Loss Pixel Control
        # 1.  Sampling replay buffer secara random
        states, rewards, actions, dones, pixel_changes = (
            experience_replay.sample_sequence(params["unroll_steps"] + 1)
        )
        # 2. Hitung loss Pixel Control
        pc_loss = local_model.control_loss(
            states, rewards, actions, dones, pixel_changes
        )

        # Hitung Loss Reward Prediction
        # 1. Sampel frame dengan peluang rewarding state = 0.5
        states, rewards, actions, dones, pixel_changes = experience_replay.sample_rp()
        # 2. Hitung loss reward prediction
        rp_loss = local_model.rp_loss(states, rewards)

        # Hitung loss Value Replay
        states, rewards, actions, dones, pixel_changes = (
            experience_replay.sample_sequence(params["unroll_steps"] + 1)
        )
        vr_loss = local_model.vr_loss(states, actions, rewards, dones)

        # Penjumlahan loss a3c, pixel control, value replay dan reward prediction
        total_loss = a3c_loss + params["pc_weight"] * pc_loss + rp_loss + vr_loss

        rewards.append(episode_rewards)

        total_loss.backward()
        # total_gradient = 0
        # for name, param in local_model.named_parameters():
        #     if param.grad is not None:
        #         # Jumlahkan absolut nilai gradien untuk semua parameter
        #         total_gradient += param.grad.abs().sum().item()
        nn.utils.clip_grad_norm_(local_model.parameters(), 0.5)
        # clipped_grad = 0
        # for name, param in local_model.named_parameters():
        #     if param.grad is not None:
        #         # Jumlahkan absolut nilai gradien untuk semua parameter
        #         clipped_grad += param.grad.abs().sum().item()
        # print()
        # print(total_gradient, clipped_grad)
        ensure_share_grads(
            local_model=local_model, global_model=global_model, device=device
        )
        optimizer.step()
        local_model._set_temperature(max(1.0, local_model.temperature * 0.9999))

    # plt.plot(
    #     np.array(rewards)
    # )
    # plt.tight_layout()
    # plt.show()
