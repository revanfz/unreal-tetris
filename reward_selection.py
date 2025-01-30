import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from model import UNREAL
from optimizer import SharedRMSprop
from replay_buffer import ReplayBuffer
from torch.distributions import Categorical
from multiprocessing.sharedctypes import SynchronizedBase
from torch import tensor, zeros, zeros_like, no_grad, from_numpy, squeeze, device
from utils import (
    make_env,
    preprocessing,
    pixel_diff,
    ensure_share_grads,
    update_progress,
)


MAX_STEP = 1_000_000
REWARD_DESIGN = ["scoring", "suriving", "heuristic"]

BETA = 0.01
GAMMA = 0.99
GRAD_NORM = 40
UNROLL_STEPS = 20
HIDDEN_SIZE = 256
LEARNING_RATE = 0.0003


def worker(
    rank: int,
    reward_design: str,
    global_model: UNREAL,
    optimizer: SharedRMSprop,
    queue: mp.Queue,
    global_steps: SynchronizedBase,
):
    env = make_env(
        id="TetrisA-v3",
        resize=84,
        level=19,
        skip=2,
        render_mode="human" if not rank else "rgb_array",
        reward_design=reward_design,
    )

    local_model = UNREAL(
        n_inputs=global_model.n_inputs,
        n_actions=global_model.n_actions,
        device=global_model.device,
        beta=BETA,
        gamma=GAMMA,
    )
    experience_replay = ReplayBuffer(2000)

    done = True
    action = F.one_hot(tensor[0].long(), num_classes=env.action_space.n)
    reward = zeros(1, 1).float()

    while not experience_replay._is_full():
        if done:
            state, _ = env.reset()
            state = preprocessing(state)
            hx = zeros(1, HIDDEN_SIZE)
            cx = zeros(1, HIDDEN_SIZE)

        with no_grad():
            state_tensor = tensor(state).float().unsqueeze(0)
            policy, _, hx, cx = local_model(
                state_tensor, prev_action, prev_reward, (hx, cx)
            )

            dist = Categorical(probs=policy)
            action = dist.sample().cpu()

        next_state, reward, done, _, _ = env.step(action.item())

        next_state = preprocessing(next_state)
        pixel_change = pixel_diff(state, next_state)
        experience_replay.store(state, reward, action.item(), done, pixel_change)

        state = next_state
        prev_action = F.one_hot(action, num_classes=env.action_space.n)
        prev_reward = tensor([[reward]]).float()

    done = True
    action = F.one_hot(tensor[0].long(), num_classes=env.action_space.n)
    reward = zeros(1, 1).float()

    while global_steps.value <= MAX_STEP:
        optimizer.zero_grad()
        local_model.load_state_dict(global_model.state_dict())

        dones = zeros(UNROLL_STEPS)
        rewards = zeros_like(dones)
        log_probs = zeros_like(dones)
        entropies = zeros_like(dones)
        values = zeros_like(dones)

        if done:
            state, info = env.reset()
            state = preprocessing(state)
            hx = zeros(1, 256)
            cx = zeros(1, 256)
            game_r = 0
        else:
            hx = hx.detach()
            cx = cx.detach()

        for step in range(UNROLL_STEPS):
            state_tensor = from_numpy(state).unsqueeze(0)
            policy, value, hx, cx = local_model(state_tensor, action, reward, (hx, cx))

            dist = Categorical(probs=policy)
            action = dist.sample()
            entropy = dist.entropy()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _, info = env.step(action.cpu().item())
            game_r += reward
            next_state = preprocessing(next_state)
            pixel_change = pixel_diff(state, next_state)
            experience_replay.store(
                state, reward, action.cpu().item(), done, pixel_change
            )

            values[step] = squeeze(value)
            entropies[step] = entropy
            log_probs[step] = squeeze(log_prob)
            dones[step] = tensor(not done)
            rewards[step] = tensor(reward)

            state = next_state
            action = F.one_hot(action, num_classes=env.action_space.n)
            reward = tensor([[reward]]).float()

            with global_steps.get_lock():
                global_steps.value += 1

            if done:
                queue.put([game_r, sum(info["statistics"].values()), info["number_of_lines"]])
                break

        # Bootstrapping
        R = 0.0
        with no_grad():
            _, R, _, _ = local_model(
                tensor(next_state).float().unsqueeze(0),
                action,
                reward,
                (hx, cx),
            )

        # Hitung loss A3C
        actor_loss, critic_loss = local_model.a3c_loss(
            rewards=rewards[: step + 1],
            R=R,
            dones=dones[: step + 1],
            log_probs=log_probs[: step + 1],
            entropies=entropies[: step + 1],
            values=values[: step + 1],
        )
        a3c_loss = actor_loss + 0.5 * critic_loss
        episode_rewards = rewards.sum().cpu().detach()

        # Hitung Loss Pixel Control
        # 1.  Sampling replay buffer secara random
        states, rewards, actions, dones, pixel_changes = (
            experience_replay.sample_sequence(UNROLL_STEPS + 1)
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
            experience_replay.sample_sequence(UNROLL_STEPS + 1)
        )
        vr_loss = local_model.vr_loss(states, actions, rewards, dones)

        # Penjumlahan loss a3c, pixel control, value replay dan reward prediction
        total_loss = a3c_loss + pc_loss + rp_loss + vr_loss

        total_loss.backward()
        nn.utils.clip_grad_norm_(local_model.parameters(), GRAD_NORM)
        ensure_share_grads(local_model=local_model, global_model=global_model)
        optimizer.step()

    queue.put(None)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    for design in REWARD_DESIGN:
        global_model = UNREAL(n_inputs=(84, 84, 3), n_actions=12, device=device("cpu"))
        global_model.share_memory()
        optimizer = SharedRMSprop(global_model.parameters(), LEARNING_RATE)

        shared_data = mp.Queue()
        global_steps = mp.Value("i", 0)

        processes = []
        all_data = []
        finished_agent = 0

        progress_process = mp.Process(
            target=update_progress,
            args=(global_steps, MAX_STEP),
            kwargs=({"desc": f"Training model with {design} reward", "unit": "steps"}),
        )
        progress_process.start()
        processes.append(progress_process)

        for rank in range(mp.cpu_count()):
            process = mp.Process(
                target=worker,
                kwargs={
                    "rank": rank,
                    "reward_design": design,
                    "global_model": global_model,
                    "optimizer": optimizer,
                    "queue": shared_data,
                    "global_steps": global_steps
                }
            )
            process.start()
            processes.append(process)

        while finished_agent < mp.cpu_count():
            data = shared_data.get()
            if data is None:
                finished_agent += 1
                continue
            all_data.append(data)

        for process in processes:
            process.join()

        data = np.array(all_data)
        df = pd.DataFrame(data, colums =["rewards", "tetriminos", "lines"])
        df.to_csv(f"{UNREAL-eval}/reward/{design}.csv", index=False)