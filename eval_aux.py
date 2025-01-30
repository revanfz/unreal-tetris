import os
import time
import torch
import argparse
import pandas as pd
import torch.nn.functional as F
import torch.multiprocessing as mp

from model import UNREAL
from optimizer import SharedRMSprop
from replay_buffer import ReplayBuffer
from torch.distributions import Categorical
from multiprocessing.synchronize import Lock
from multiprocessing.managers import DictProxy
from multiprocessing.sharedctypes import Synchronized
from utils import ensure_share_grads, make_env, preprocessing, pixel_diff, update_progress


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

DATA_DIR = "./UNREAL-eval/auxiliary/"
postfix_pc = f"{'no-' if not params.pixel_control else ''}pc"
postfix_rp = f"{'no-' if not params.reward_prediction else ''}rp"
postfix_vr = f"{'no-' if not params.value_replay else ''}vr"
filename = f"UNREAL {postfix_pc}_{postfix_rp}_{postfix_vr}"

# HYPERPARAMETER
MAX_STEPS = 1_000_000
LEARNING_RATE = 0.00012
BETA = 0.00318
GAMMA = 0.95
UNROLL_STEPS = 20
HIDDEN_SIZE = 256
PC_WEIGHT = 0.05478

def store_data(shared_dict: DictProxy, data: dict, lock: Lock, type: str):
    if type == "game":
        with lock:
            # Dapatkan data saat ini
            current_lines = list(shared_dict["lines"])
            current_lines.append(data["lines"])
            shared_dict["lines"] = current_lines


            current_blocks = list(shared_dict["block_placed"]) 
            current_blocks.append(data["block_placed"])
            shared_dict["block_placed"] = current_blocks
 
    with lock:
        current_scores = list(shared_dict["score"])
        current_rewards = list(shared_dict["rewards"]) 
        current_times = list(shared_dict["episode_time"])
        current_lengths = list(shared_dict["episode_length"])
        
        current_scores.append(data["score"])
        current_rewards.append(data["rewards"])
        current_times.append(data["episode_time"])
        current_lengths.append(data["episode_length"])
        
        shared_dict["score"] = current_scores
        shared_dict["rewards"] = current_rewards
        shared_dict["episode_time"] = current_times
        shared_dict["episode_length"] = current_lengths


def agent(rank: int, global_model: UNREAL, global_steps: Synchronized, shared_episode_dict: DictProxy, shared_game_dict, lock: Lock):
    device = torch.device("cpu")

    model =  UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=6,
        hidden_size=HIDDEN_SIZE,
        beta=BETA,
        gamma=GAMMA,
        device=device,
        pc=global_model.use_pc,
        rp=global_model.use_rp,
        vr=global_model.use_vr
    )
    model.train()
    optimizer = SharedRMSprop(params=model.parameters(), lr=LEARNING_RATE)

    env = make_env(
        resize=84,
        level = 19,
        skip=2,
        id="TetrisA-v3",
        render_mode="human" if not rank else "rgb_array",
        record_statistics=True,
        num_games=1
    )

    done = True

    experience_replay = ReplayBuffer(2000)
    state, info = env.reset()
    state = preprocessing(state)
    prev_action = F.one_hot(torch.LongTensor([0]), env.action_space.n).to(device)
    prev_reward = torch.zeros(1, 1).to(device)
    hx = torch.zeros(1, HIDDEN_SIZE).to(device)
    cx = torch.zeros(1, HIDDEN_SIZE).to(device)

    # while not experience_replay._is_full():
    for i in range(500):
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
    action = F.one_hot(torch.tensor([0]).long(), env.action_space.n).to(device)
    reward = torch.zeros(1, 1, device=device)

    while global_steps.value < MAX_STEPS:
        optimizer.zero_grad()
        model.load_state_dict(global_model.state_dict())

        dones = torch.zeros(UNROLL_STEPS, device=device)
        rewards = torch.zeros_like(dones, device=device)
        log_probs = torch.zeros_like(dones, device=device)
        entropies = torch.zeros_like(dones, device=device)
        values = torch.zeros_like(dones, device=device)

        episode_rewards = 0
        
        if done:
            state, info = env.reset(seed=42+rank)
            state = preprocessing(state)
            hx = torch.zeros(1, HIDDEN_SIZE, device=device)
            cx = torch.zeros(1, HIDDEN_SIZE, device=device)
            game_reward = 0
        else:
            hx = hx.detach()
            cx = cx.detach()

        for step in range(UNROLL_STEPS):
            episode_time = time.perf_counter()
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            policy, value, hx, cx = model(
                state_tensor, action, reward, (hx, cx)
            )

            dist = Categorical(probs=policy)
            action = dist.sample()
            entropy = dist.entropy()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _, info = env.step(action.cpu().item())
            game_reward += reward
            episode_rewards += reward
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
            reward = torch.tensor([[reward]], device=device).float()

            with global_steps.get_lock():
                global_steps.value += 1

            if done:
                store_data(shared_game_dict, {
                    "lines": info['number_of_lines'],
                    "block_placed": sum(info["statistics"].values()),
                    "score": info['score'],
                    "rewards": game_reward,
                    "episode_time": info["episode"]["t"],
                    "episode_length": info["episode"]["l"],
                }, lock, "game")
                break
        
        store_data(shared_episode_dict, {
            "score": info['score'],
            "rewards": episode_rewards,
            "episode_time": time.perf_counter() - episode_time,
            "episode_length": step + 1,
        }, lock, "episode")

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
        actor_loss, critic_loss = model.a3c_loss(
            rewards=rewards[: step + 1],
            R=R,
            dones=dones[: step + 1],
            log_probs=log_probs[: step + 1],
            entropies=entropies[: step + 1],
            values=values[: step + 1],
        )
        a3c_loss = actor_loss + 0.5 * critic_loss
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

        if model.use_rp:
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
        ensure_share_grads(
            local_model=model, global_model=global_model
        )
        optimizer.step()


if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    global_model =  UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=6,
        hidden_size=HIDDEN_SIZE,
        beta=BETA,
        gamma=GAMMA,
        device=torch.device("cpu"),
        pc=params.pixel_control,
        rp=params.reward_prediction,
        vr=params.value_replay
    )
    global_model.share_memory()

    manager = mp.Manager()
    lock = mp.Lock()

    shared_episode_dict = manager.dict({
        "score": manager.list(),
        "rewards": manager.list(),
        "episode_time": manager.list(),
        "episode_length": manager.list()
    })

    shared_game_dict = manager.dict({
        "lines": manager.list(),
        "score": manager.list(),
        "rewards": manager.list(),
        "block_placed": manager.list(),
        "episode_time": manager.list(),
        "episode_length": manager.list()
    })

    global_steps = mp.Value("i", 0)
    processes = []

    progress_process = mp.Process(
        target=update_progress,
        args=(
            global_steps,
            MAX_STEPS
        ),
        kwargs=({"desc": f"{filename}"})
    )
    progress_process.start()
    processes.append(progress_process)

    for rank in range(mp.cpu_count()):
        process = mp.Process(
            target=agent,
            args=(
                rank,
                global_model,
                global_steps,
                shared_episode_dict,
                shared_game_dict,
                lock
            )
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    df_game = pd.DataFrame(dict(shared_game_dict))
    df_episode = pd.DataFrame(dict(shared_episode_dict))
    df_game.to_csv(f"{DATA_DIR}/GAME-{filename}.csv", index=False)
    df_episode.to_csv(f"{DATA_DIR}/EPISODE-{filename}.csv", index=False)

    torch.save(global_model.state_dict(), f"{DATA_DIR}/{filename}.pt")