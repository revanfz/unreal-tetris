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
    beta=0.00067,
    gamma=0.95,
    hidden_size=256,
    task_weight=0.01,
)

device = torch.device("cuda")

if __name__ == "__main__":
    env = make_env(resize=84, render_mode="human", level=19, skip=2)

    global_model = UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=env.action_space.n,
        hidden_size=256,
        device=torch.device("cpu"),
        beta=params["beta"],
        gamma=params["gamma"],
    )
    optimizer = SharedRMSprop(global_model.parameters(), params["lr"])
    local_model = UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=env.action_space.n,
        hidden_size=256,
        device=device,
    )
    local_model.train()
    experience_replay = ReplayBuffer(2000)

    state, info = env.reset()
    state = preprocessing(state)
    prev_action = F.one_hot(torch.LongTensor([0]), env.action_space.n).to(device)
    prev_reward = torch.zeros(1, 1).to(device)
    hx = torch.zeros(1, params["hidden_size"]).to(device)
    cx = torch.zeros(1, params["hidden_size"]).to(device)

    while not experience_replay._is_full():
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

    while True:
        optimizer.zero_grad()
        local_model.load_state_dict(global_model.state_dict())
        eps_length = 0

        if done:
            state, info = env.reset()
            state = preprocessing(state)
            action = F.one_hot(torch.LongTensor([0]), num_classes=env.action_space.n).to(device)
            reward = torch.zeros(1, 1).to(device)
            hx = torch.zeros(1, params["hidden_size"]).to(device)
            cx = torch.zeros(1, params["hidden_size"]).to(device)
            eps_r = []
        else:
            hx = hx.detach()
            cx = cx.detach()

        for t in range(params["unroll_steps"]):
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            policy, value, hx, cx = local_model(state_tensor, action, reward, (hx, cx))

            probs = F.softmax(policy, dim=1)
            dist = Categorical(probs)
            action = dist.sample()

            action = probs.argmax().unsqueeze(0)


            next_state, reward, done, _, info = env.step(action.item())
            if reward:
                eps_r.append(reward)
            next_state = preprocessing(next_state)
            pixel_change = pixel_diff(state, next_state)
            experience_replay.store(state, reward, action.item(), done, pixel_change)
            state = next_state

            action = F.one_hot(action, num_classes=env.action_space.n).to(device)
            reward = torch.FloatTensor([[reward]]).to(device)

            eps_length += 1

            if done:
                print("Episode rewards: ", eps_r)
                break

        # Bootstrapping
        R = 0.0
        if not done:
            with torch.no_grad():
                _, R, _, _ = local_model(
                    torch.FloatTensor(next_state).unsqueeze(0).to(device),
                    action,
                    reward,
                    (hx, cx),
                )

        states, rewards, actions, dones, pixel_change = experience_replay.sample(eps_length)
        a3c_loss, entropy = local_model.a3c_loss(
            states=states,
            dones=dones,
            actions=actions,
            rewards=rewards,
            R=R,
        )

        # Pixel Control
        # states, rewards, actions, dones, pcs = experience_replay.sample(name="pc")
        states, rewards, actions, dones, pixel_changes = (
            experience_replay.sample_sequence(params["unroll_steps"] + 1)
        )
        pc_loss = local_model.control_loss(
            states, rewards, actions, dones, pixel_changes
        )

        # Reward Prediction
        states, rewards, actions, dones, pixel_changes = (
            experience_replay.sample_rp()
        )
        rp_loss = local_model.rp_loss(states, rewards)

        # Value replay
        states, rewards, actions, dones, pixel_changes = (
            experience_replay.sample_sequence(params["unroll_steps"] + 1)
        )
        vr_loss = local_model.vr_loss(states, actions, rewards, dones)

        # print(f"A3C Loss = {a3c_loss}\t PC Loss = {pc_loss}\t VR Loss = {vr_loss}\t RP Loss = {rp_loss}\n" )
        # print(f"Entropy: {entropy.mean().item()}")
        total_loss = a3c_loss + params["task_weight"] * pc_loss + rp_loss + vr_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40)
        ensure_share_grads(
            local_model=local_model, global_model=global_model, device=device
        )
        optimizer.step()