import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import torch
import optuna
import pickle
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from tqdm import tqdm
from typing import Union
from model import UNREAL
from replay_buffer import ReplayBuffer
from torch.distributions import Categorical
from multiprocessing.synchronize import Event
from optimizer import SharedAdam, SharedRMSprop
from multiprocessing.sharedctypes import Synchronized
from utils import ensure_share_grads, make_env, pixel_diff, preprocessing, update_progress


def print_best_trial(trials, index: int, metric_name: str):
    best_trial = max(trials, key=lambda t: t.values[index])
    print(f"\nTrial with {'highest' if metric_name != 'game tries' else 'lowest'} {metric_name}:")
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
    global_scores: Synchronized,
    global_blocks: Synchronized,
    global_lines: Synchronized,
    global_tries: Synchronized,
    # stop_event: Event,
    # trial: optuna.Trial,
) -> None:
    try:
        device = params["device"]
        torch.manual_seed(42 + rank)
        env = make_env(resize=84, render_mode="rgb_array", id="TetrisA-v2", level=19)

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

        state, info = env.reset()
        state = preprocessing(state)
        action = F.one_hot(torch.LongTensor([0]), env.action_space.n).to(device)
        reward = torch.zeros(1, 1).to(device)

        while not experience_replay._is_full():
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                policy, value, _, _ = local_model(
                    state_tensor, action, reward, None
                )
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
                state, info = env.reset()
                state = preprocessing(state)

        done = True
        action = F.one_hot(torch.LongTensor([0]), num_classes=env.action_space.n).to(device)
        reward = torch.zeros(1, 1).to(device)

        # while not stop_event.is_set():
        while global_steps.value <= params["max_steps"]:
            optimizer.zero_grad()
            local_model.load_state_dict(global_model.state_dict())
            eps_length = 0

            if done:
                state, info = env.reset()
                state = preprocessing(state)
                hx = torch.zeros(1, params["hidden_size"]).to(device)
                cx = torch.zeros(1, params["hidden_size"]).to(device)
            else:
                hx = hx.detach()
                cx = cx.detach()

            for _ in range(params["unroll_steps"]):
                with global_steps.get_lock():
                    global_steps.value += 1

                state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                policy, value, hx, cx = local_model(state_tensor, action, reward, (hx, cx))

                dist = Categorical(probs=policy)
                action = dist.sample()

                next_state, reward, done, _, info = env.step(action.item())
                next_state = preprocessing(next_state)
                pixel_change = pixel_diff(state, next_state)
                experience_replay.store(state, reward, action.item(), done, pixel_change)
                state = next_state

                action = F.one_hot(action, num_classes=env.action_space.n).to(device)
                reward = torch.FloatTensor([[reward]]).to(device)

                eps_length += 1

                if done:
                    with global_scores.get_lock():
                        global_scores.value += info["score"]

                    with global_blocks.get_lock():
                        global_blocks.value += sum(info["statistics"].values())

                    with global_lines.get_lock():
                        global_lines.value += info["number_of_lines"]

                    with global_tries.get_lock():
                        global_tries.value += 1
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

            # Hitung loss A3C
            states, rewards, actions, dones, pixel_change = experience_replay.sample(eps_length)
            a3c_loss, entropy = local_model.a3c_loss(
                states=states,
                dones=dones,
                actions=actions,
                rewards=rewards,
                R=R,
            )

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

            # print(f"A3C Loss = {a3c_loss}\t PC Loss = {pc_loss}\t VR Loss = {vr_loss}\t RP Loss = {rp_loss}\n" )

            # Penjumlahan loss a3c, pixel control, value replay dan reward prediction
            total_loss = (
                a3c_loss + params["pc_weight"] * pc_loss + rp_loss + vr_loss 
            )

            total_loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), 0.5)
            ensure_share_grads(
                local_model=local_model, global_model=global_model, device=device
            )
            optimizer.step()
                
        print(f"Agent {rank} training process finished.")

    except (KeyboardInterrupt, torch.multiprocessing.ProcessError) as e:
        print(e)

    finally:
        env.close()


def objective(trial: optuna.Trial):
    try:
        env = make_env(resize=84, render_mode="rgb_array", level=19, skip=2)

        params = {
            "lr": trial.suggest_float("learning rate", 1e-4, 5e-3, log=True),
            "pc_weight": trial.suggest_float("lambda pc", 0.01, 0.1, log=True),
            "beta": trial.suggest_float("entropy coefficient", 5e-4, 1e-2, log=True),
            "gamma": 0.99,
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
        # stop_event = mp.Event()
        global_scores = mp.Value("i", 0)
        global_blocks = mp.Value("i", 0)
        global_lines = mp.Value("i", 0)
        global_tries = mp.Value("i", 0)
        global_steps = mp.Value("i", 0)
        # start_time = time.time()

        progress_process = mp.Process(
            target=update_progress,
            args=(
                global_steps,
                params["max_steps"],
            ),
            kwargs=({"desc": "Total Steps"})
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
                    global_scores,
                    global_blocks,
                    global_lines,
                    global_tries,
                    # stop_event,
                    # trial,
                ),
            )
            p.start()
            processes.append(p)

        # train_time = 3600
        # with tqdm(total=train_time, desc=f"Trial {trial.number}", unit="s") as pbar:
        #     pbar.update(int(time.time() - start_time))
        #     while time.time() - start_time < train_time:
        #         if all(not p.is_alive() for p in processes):
        #             break
        #         time.sleep(1)
        #         pbar.update(1)

        #         # Cek apakah trial harus di-prune
        #         # if trial.should_prune():
        #         #     stop_event.set()
        #         #     break

        # stop_event.set()

        for process in processes:
            process.join()
            # process.join(timeout=10)
            # if process.is_alive():
            #     process.terminate()

        # mean_rewards = global_scores.value
        return global_scores.value, global_blocks.value, global_lines.value, global_tries.value


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
            url="sqlite:///tuning/hpo-UNREAL.db",
            engine_kwargs={"connect_args": {"timeout": 30}},
        )
        study = optuna.create_study(
            study_name="final",
            storage=storage,
            load_if_exists=True,
            directions=["maximize", "maximize", "maximize", "minimize"]
        )

        if os.path.isfile("./tuning/sampler.pkl"):
            restored_sampler = pickle.load(open("tuning/sampler.pkl", "rb"))
            study.sampler = restored_sampler

        completed_trials = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        n_trials = 45

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # print("Number of finished trials: ", len(study.trials))

        # print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

        metrics = [
            (0, "scores"),
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
