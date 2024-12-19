import os

os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import torch
import optuna
import pickle
import logging
import builtins
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from tqdm import tqdm
from model import UNREAL
from typing import Union
from replay_buffer import ReplayBuffer
from torch.distributions import Categorical
from multiprocessing.synchronize import Event
from optimizer import SharedAdam, SharedRMSprop
from multiprocessing.sharedctypes import Synchronized
from utils import (
    ensure_share_grads,
    make_env,
    pixel_diff,
    preprocessing,
    update_progress,
)


def print_best_trial(trials, index: int, metric_name: str):
    func = "max" if metric_name != "game tries" else "min"
    best_func = getattr(builtins, func)
    best_trial = best_func(trials, key=lambda t: t.values[index])
    print(
        f"\nTrial with {'highest' if metric_name != 'game tries' else 'lowest'} {metric_name}:"
    )
    print(f"\tnumber: {best_trial.number}")
    print(f"\tparams: {best_trial.params}")
    print(f"\tvalues: {best_trial.values}")
    print()


def train(
    rank: int,
    params: dict,
    global_model: UNREAL,
    optimizer: Union[SharedAdam, SharedRMSprop],
    global_steps: Synchronized,
    global_rewards: Synchronized,
    global_blocks: Synchronized,
    global_lines: Synchronized,
    global_tries: Synchronized,
) -> None:
    try:
        device = params["device"]
        torch.manual_seed(42 + rank)
        env = make_env(
            resize=84,
            render_mode="rgb_array",
            id="TetrisA-v3",
            level=19,
            skip=2
        )

        local_model = UNREAL(
            n_inputs=(84, 84, 3),
            n_actions=env.action_space.n,
            hidden_size=params["hidden_size"],
            beta=params["beta"],
            gamma=params["gamma"],
            device=device,
        )
        local_model.train()
        experience_replay = ReplayBuffer(2000)

        state, info = env.reset(seed=42 + rank)
        state = preprocessing(state)
        action = F.one_hot(torch.LongTensor([0]), env.action_space.n).to(device)
        reward = torch.zeros(1, 1).to(device)

        while not experience_replay._is_full():
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                policy, value, _, _ = local_model(state_tensor, action, reward, None)
                probs = F.softmax(policy, dim=1)
                dist = Categorical(probs=probs)
                action = dist.sample().cpu()

            next_state, reward, done, _, info = env.step(action.item())
            next_state = preprocessing(next_state)
            pixel_change = pixel_diff(state, next_state)
            experience_replay.store(state, reward, action.item(), done, pixel_change)
            state = next_state
            action = F.one_hot(action, num_classes=env.action_space.n).to(device)
            reward = torch.FloatTensor([[reward]]).to(device)

            if done:
                state, info = env.reset(seed=42 + rank)
                state = preprocessing(state)

        done = True
        action = F.one_hot(torch.LongTensor([0]), num_classes=env.action_space.n).to(
            device
        )
        reward = torch.zeros(1, 1).to(device)

        # while not stop_event.is_set():
        while global_steps.value <= params["max_steps"]:
            optimizer.zero_grad()
            local_model.load_state_dict(global_model.state_dict())

            dones = torch.zeros(params["unroll_steps"], device=device)
            rewards = torch.zeros_like(dones, device=device)
            log_probs = torch.zeros_like(dones, device=device)
            entropies = torch.zeros_like(dones, device=device)
            values = torch.zeros_like(dones, device=device)

            if done:
                state, info = env.reset(seed=42 + rank)
                state = preprocessing(state)
                hx = torch.zeros(1, params["hidden_size"]).to(device)
                cx = torch.zeros(1, params["hidden_size"]).to(device)
                episode_rewards = 0
            else:
                hx = hx.detach()
                cx = cx.detach()

            for step in range(params["unroll_steps"]):
                with global_steps.get_lock():
                    global_steps.value += 1

                state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                policy, value, hx, cx = local_model(
                    state_tensor, action, reward, (hx, cx)
                )

                dist = Categorical(probs=policy)
                action = dist.sample()
                entropy = dist.entropy()
                log_prob = dist.log_prob(action)

                next_state, reward, done, _, info = env.step(action.cpu().item())
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

                if done:
                    with global_rewards.get_lock():
                        global_rewards.value += episode_rewards

                    with global_blocks.get_lock():
                        global_blocks.value += sum(info["statistics"].values())

                    with global_lines.get_lock():
                        global_lines.value += info["number_of_lines"]

                    with global_tries.get_lock():
                        global_tries.value += 1
                    break

            # Bootstrapping
            R = 0.0
            with torch.no_grad():
                _, R, _, _ = local_model(
                    torch.tensor(next_state, device=device).float().unsqueeze(0),
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
            states, rewards, actions, dones, pixel_changes = (
                experience_replay.sample_rp()
            )
            # 2. Hitung loss reward prediction
            rp_loss = local_model.rp_loss(states, rewards)

            # Hitung loss Value Replay
            states, rewards, actions, dones, pixel_changes = (
                experience_replay.sample_sequence(params["unroll_steps"] + 1)
            )
            vr_loss = local_model.vr_loss(states, actions, rewards, dones)

            # Penjumlahan loss a3c, pixel control, value replay dan reward prediction
            total_loss = a3c_loss + params["pc_weight"] * pc_loss + rp_loss + vr_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), params["grad_norm"])
            ensure_share_grads(
                local_model=local_model, global_model=global_model,
            )
            optimizer.step()

        print(f"Agent {rank} training process finished.")

    except (KeyboardInterrupt, torch.multiprocessing.ProcessError) as e:
        print(e)

    finally:
        env.close()


def objective(trial: optuna.Trial):
    try:
        env = make_env(
            resize=84,
            render_mode="rgb_array",
            level=19,
            skip=4,
        )

        params = {
            "lr": trial.suggest_float("learning rate", 1e-4, 5e-3, log=True),
            "pc_weight": trial.suggest_float("lambda pc", 0.01, 0.1, log=True),
            "beta": trial.suggest_float("entropy coefficient", 5e-4, 1e-2, log=True),
            "gamma": trial.suggest_categorical("discount factor", [0.95, 0.99]),
            "grad_norm": trial.suggest_categorical("gradient clipping", [0.5, 40]),
            # "gamma": 0.99,
            "optimizer": "RMSProp",
            "device": torch.device("cpu"),
            "hidden_size": 256,
            "n_actions": env.action_space.n,
            "model_path": "trained_models",
            "input_shape": (84, 84, 3),
            "unroll_steps": 20,
            "max_steps": 100_000,
        }

        del env

        global_model = UNREAL(
            n_inputs=params["input_shape"],
            n_actions=params["n_actions"],
            hidden_size=params["hidden_size"],
            beta=params["beta"],
            gamma=params["gamma"],
            device=torch.device("cpu"),
        )
        global_model.share_memory()

        if params["optimizer"] == "Adam":
            optimizer = SharedAdam(global_model.parameters(), lr=params["lr"])
        elif params["optimizer"] == "RMSProp":
            optimizer = SharedRMSprop(global_model.parameters(), lr=params["lr"])

        processes = []
        global_rewards = mp.Value("f", 0)
        global_blocks = mp.Value("i", 0)
        global_lines = mp.Value("i", 0)
        global_tries = mp.Value("i", 0)
        global_steps = mp.Value("i", 0)

        progress_process = mp.Process(
            target=update_progress,
            args=(
                global_steps,
                params["max_steps"],
            ),
            kwargs=({"desc": "Total Steps", "unit": "steps"}),
        )
        progress_process.start()
        processes.append(progress_process)

        for rank in range(mp.cpu_count()):
            p = mp.Process(
                target=train,
                args=(
                    rank,
                    params,
                    global_model,
                    optimizer,
                    global_steps,
                    global_rewards,
                    global_blocks,
                    global_lines,
                    global_tries,
                ),
            )
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        return (
            global_rewards.value,
            global_blocks.value / global_tries.value,
            global_lines.value,
            global_tries.value,
        )

    except KeyboardInterrupt:
        raise KeyboardInterrupt("Tuning dihentikan.")

    except Exception as e:
        raise Exception(f"Error {e}")


if __name__ == "__main__":
    try:
        optuna.logging.get_logger("optuna").addHandler(
            logging.StreamHandler(sys.stdout)
        )
        storage = optuna.storages.RDBStorage(
            url="sqlite:///tuning/UNREAL.db",
            engine_kwargs={"connect_args": {"timeout": 30}},
        )
        study = optuna.create_study(
            study_name="UNREAL",
            storage=storage,
            load_if_exists=True,
            directions=["maximize", "maximize", "maximize", "minimize"],  # UNREAL
        )

        if os.path.isfile("./tuning/sampler.pkl"):
            restored_sampler = pickle.load(open("tuning/sampler.pkl", "rb"))
            study.sampler = restored_sampler

        n_trials = 20

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        metrics = [
            (0, "rewards"),
            (1, "blocks placed"),
            (2, "lines cleared"),
            (3, "game tries"),
        ]

        for index, metric_name in metrics:
            print_best_trial(study.best_trials, index, metric_name)

        print("Tuning selesai.")

    except (KeyboardInterrupt, optuna.exceptions.OptunaError, Exception) as e:
        print(f"Error: {e}")
        print("Tuning berhenti.")

    finally:
        with open("./tuning/sampler.pkl", "wb") as fout:
            pickle.dump(study.sampler, fout)
        print("Proses tuning dihentikan")
