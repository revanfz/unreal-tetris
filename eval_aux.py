import os
import time
import torch
import argparse
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from model import UNREAL
from optimizer import SharedRMSprop
from replay_buffer import ReplayBuffer
from torch.distributions import Categorical
from utils import make_env, preprocessing, pixel_diff


def get_args():
    parser = argparse.ArgumentParser(
        """
            Evaluasi model UNREAL: 
            IMPLEMENTASI ARSITEKTUR UNSUPERVISED REINFORCEMENT WITH AUXILIARY LEARNING (UNREAL)
            UNTUK MENGHASILKAN AGEN CERDAS (STUDI KASUS: PERMAINAN TETRIS)
        """
    )
    parser.add_argument(
        "--start-case", type=int, default=1, help="Starting point test case"
    )
    parser.add_argument(
        "--pixel-control", default=True, action=argparse.BooleanOptionalAction, help="Menggunakan pixel control atau tidak"
    )
    parser.add_argument(
        "--reward-prediction", default=True, action=argparse.BooleanOptionalAction, help="Menggunakan reward prediction atau tidak"
    )
    parser.add_argument(
        "--value-replay", default=True, action=argparse.BooleanOptionalAction, help="Menggunakan value replay atau tidak"
    )
    args = parser.parse_args()
    return args

params = get_args()

DATA_DIR = "./UNREAL-tetris/aux/"
postfix_pc = f"{'no-' if not params.pixel_control else ''}pc"
postfix_rp = f"{'no-' if not params.reward_prediction else ''}rp"
postfix_vr = f"{'no-' if not params.value_replay else ''}vr"
filename = f"UNREAL {postfix_pc}_{postfix_rp}_{postfix_vr}"

# HYPERPARAMETER
MAX_STEPS = 100_000
LEARNING_RATE = 0.0002
BETA = 0.00102
GAMMA = 0.99
UNROLL_STEPS = 20
HIDDEN_SIZE = 256
PC_WEIGHT = 0.08928


if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    device = torch.device("cpu")

    model =  UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=12,
        hidden_size=HIDDEN_SIZE,
        beta=BETA,
        gamma=GAMMA,
        device=device,
        pc=params.pixel_control,
        rp=params.reward_prediction,
        vr=params.value_replay
    )
    model.train()
    optimizer = SharedRMSprop(params=model.parameters(), lr=LEARNING_RATE)

    env = make_env(
        resize=84,
        level = 19,
        skip=2,
        id="TetrisA-v2"
    )

    data_per_episode = {
        "score": [],
        "rewards": [],
        "episode_time": [],
        "episode_length": [],
    }

    data_per_game = {
        "lines": [],
        "score": [],
        "rewards": [],
        "block_placed": [],
        "episode_time": [],
        "episode_length": [],
    }

    done = True

    if params.value_replay or params.pixel_control or params.reward_predicion:
        experience_replay = ReplayBuffer(2000)
        state, info = env.reset()
        state = preprocessing(state)
        prev_action = F.one_hot(torch.LongTensor([0]), env.action_space.n).to(device)
        prev_reward = torch.zeros(1, 1).to(device)
        hx = torch.zeros(1, HIDDEN_SIZE).to(device)
        cx = torch.zeros(1, HIDDEN_SIZE).to(device)

        while not experience_replay._is_full():
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                policy, _, _, _ = model(
                    state_tensor, prev_action, prev_reward, None
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
                hx = torch.zeros(1, HIDDEN_SIZE).to(device)
                cx = torch.zeros(1, HIDDEN_SIZE).to(device)

    done = True

    for step in tqdm(range(MAX_STEPS), desc=f"Evaluating {filename}"):
        for t in range(UNROLL_STEPS):
            episode_rewards = 0
            episode_time = time.perf_counter()
            if done:
                start_time = time.perf_counter()
                episode_length = 0
                game_reward = 0
                state, info = env.reset()
                state = preprocessing(state)
                action = F.one_hot(torch.LongTensor([0]), num_classes=env.action_space.n).to(device)
                reward = torch.zeros(1, 1).to(device)
                hx = torch.zeros(1, 256).to(device)
                cx = torch.zeros(1, 256).to(device)
            else:
                hx = hx.detach()
                cx = cx.detach()

            state_tensor =  torch.from_numpy(state).unsqueeze(0).to(device)
            policy, _, hx, cx = model(
                state_tensor, action, reward, (hx, cx)
            )

            dist = Categorical(probs=policy)
            action = dist.sample()

            next_state, reward, done, _, info = env.step(action.item())
            game_reward += reward
            episode_rewards += reward
            if done:
                stop_time = time.perf_counter()
            next_state = preprocessing(next_state)
            pixel_change = pixel_diff(state, next_state)
            experience_replay.store(
                state, reward, action.item(), done, pixel_change
            )
            state = next_state

            action = F.one_hot(action, num_classes=env.action_space.n).to(
                device
            )
            reward = torch.FloatTensor([[reward]]).to(device)

            episode_length += 1

            if done:
                data_per_game["lines"].append(info['number_of_lines'])
                data_per_game["block_placed"].append(sum(info["statistics"].values()))
                data_per_game["score"].append(info['score'])
                data_per_game["rewards"].append(game_reward)
                data_per_game["episode_time"].append(stop_time - start_time)
                data_per_game["episode_length"].append(episode_length)
                break
        
        data_per_episode["score"].append(info['score'])
        data_per_episode["rewards"].append(episode_rewards)
        data_per_episode["episode_time"].append(time.perf_counter() - episode_time)
        data_per_episode["episode_length"].append(t + 1)

        # Bootstrapping
        R = 0.0
        if not done:
            with torch.no_grad():
                _, R, _, _ = model(
                    torch.FloatTensor(next_state).unsqueeze(0).to(device),
                    action,
                    reward,
                    (hx, cx),
                )

        # Hitung loss A3C
        states, rewards, actions, dones, pixel_change = experience_replay.sample(episode_length)
        a3c_loss, entropy = model.a3c_loss(
            states=states,
            dones=dones,
            actions=actions,
            rewards=rewards,
            R=R,
        )
        total_loss = a3c_loss

        if model.use_pc:
            # Hitung Loss Pixel Control
            # 1.  Sampling replay buffer secara random
            states, rewards, actions, dones, pixel_changes = (
                experience_replay.sample_sequence(UNROLL_STEPS + 1)
            )
            # 2. Hitung loss Pixel Control
            pc_loss = model.control_loss(
                states, rewards, actions, dones, pixel_changes
            )

            total_loss += pc_loss

        if model.use_pc:
            # Hitung Loss Reward Prediction
            # 1. Sampel frame dengan peluang rewarding state = 0.5
            states, rewards, actions, dones, pixel_changes = (
                experience_replay.sample_rp()
            )
            # 2. Hitung loss reward prediction
            rp_loss = model.rp_loss(states, rewards)

            total_loss += rp_loss

        if model.use_vr:
            # Hitung loss Value Replay
            states, rewards, actions, dones, pixel_changes = (
                experience_replay.sample_sequence(UNROLL_STEPS + 1)
            )
            vr_loss = model.vr_loss(states, actions, rewards, dones)
            total_loss += vr_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
        optimizer.step()

    df_game = pd.DataFrame(data_per_game)
    df_episode = pd.DataFrame(data_per_episode)
    df_game.to_csv(f"{DATA_DIR}/GAME-{filename}.csv", index=False)
    df_episode.to_csv(f"{DATA_DIR}/EPISODE-{filename}.csv", index=False)
