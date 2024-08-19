import os

os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import torch
import optuna
import logging
import gym_tetris
import torch.nn.functional as F
import torch.multiprocessing as mp

from optimizer import SharedAdam
from model import ActorCriticLSTM
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
from multiprocessing.sharedctypes import Synchronized
from gym.wrappers import FrameStack, GrayScaleObservation
from utils import ensure_share_grads, preprocess_frame_stack, update_progress

metrics = [
    (0, "reward"),
    (1, "blocks placed"),
    (2, "lines cleared"),
    (3, "episodes length"),
]


def save_model(
    global_model: ActorCriticLSTM,
    optimizer: SharedAdam,
    trial_number: int,
    params: dict,
):
    data = {
        "model_state_dict": global_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    for key, value in params.items():
        data[key] = value

    torch.save(
        data,
        "trained_models/tuning_a3c_tetris-{}.tar".format(trial_number),
    )


def print_best_trial(trials, index: int, metric_name: str):
    best_trial = max(trials, key=lambda t: t.values[index])
    print(f"Trial with highest {metric_name}:")
    print(f"\tnumber: {best_trial.number}")
    print(f"\tparams: {best_trial.params}")
    print(f"\tvalues: {best_trial.values}")
    print()

def a3c_train(
    rank: int,
    opt: dict,
    global_model: ActorCriticLSTM,
    optimizer: SharedAdam,
    global_episodes: Synchronized,
    # global_steps: Synchronized,
) -> None:
    try:
        device = opt["device"]

        torch.manual_seed(42 + rank)
        env = gym_tetris.make("TetrisA-v3", apply_api_compatibility=True)
        env = JoypadSpace(env, MOVEMENT)
        env = GrayScaleObservation(env)
        env = FrameStack(env, 4)

        local_model = ActorCriticLSTM(
            opt["input_shape"], opt["n_actions"], opt["hidden_size"]
        ).to(device)
        local_model.train()

        done = True

        # while global_steps.value <= opt["max_steps"]:
        while global_episodes.value <= opt["max_episodes"]:
            optimizer.zero_grad()
            local_model.load_state_dict(global_model.state_dict())

            if done:
                state, _ = env.reset()
                hx = torch.zeros(1, opt["hidden_size"]).to(device)
                cx = torch.zeros(1, opt["hidden_size"]).to(device)
            else:
                hx = hx.data
                cx = cx.data

            episode_rewards = 0
            values, log_probs, rewards, entropies = [], [], [], []

            for step in range(opt["minibatch_size"]):
                state = preprocess_frame_stack(state).to(device)
                policy, value, hx, cx = local_model(state.unsqueeze(0), hx, cx)

                probs = F.softmax(policy, dim=1)
                log_prob = F.log_softmax(policy, dim=1)
                entropy = -(log_prob * probs).sum(1)
                action = probs.multinomial(1).data
                log_prob = log_prob.gather(1, action)

                state, reward, done, _, _ = env.step(action.item())
                episode_rewards += reward

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)

                if done:
                    break

            R = torch.zeros(1, 1).to(device)
            gae = torch.zeros(1, 1).to(device)

            if not done:
                bootstrap_state = preprocess_frame_stack(state).to(device)
                _, value, _, _ = local_model(bootstrap_state.unsqueeze(0), hx, cx)
                R = value.detach()
            values.append(R)

            actor_loss = 0
            critic_loss = 0

            for t in reversed(range(len(rewards))):
                R = opt["gamma"] * R + rewards[t]
                advantage = R - values[t]
                critic_loss += 0.5 * advantage.pow(2)

                # GAE
                delta_t = (
                    rewards[t] + opt["gamma"] * values[t + 1].data - values[t].data
                )
                gae = gae * opt["gamma"] + delta_t
                actor_loss -= (log_probs[t] * gae) - (opt["beta"] * entropies[t])

            total_loss = actor_loss + 0.5 * critic_loss
            total_loss.backward()

            if opt["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)

            ensure_share_grads(
                local_model=local_model, global_model=global_model, device=device
            )
            optimizer.step()

            with global_episodes.get_lock():
                global_episodes.value += 1

            # print(f"Agent {rank} finished.\tGlobal episodes: {global_episodes.value}")

        print(f"Agent {rank} training process finished.")

    except (KeyboardInterrupt, torch.multiprocessing.ProcessError) as e:
        print(e)

    finally:
        env.close()


def objective(trial: optuna.Trial):
    lr_values = [5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    beta_values = [0.1, 0.05, 0.01, 0.005, 0.001]
    # "lr": trial.suggest_categorical("learning_rate", lr_values),

    params = {
        # "lr": trial.suggest_float("lr", 1e-7, 1e-2, log=True),
        # "beta": trial.suggest_float("beta", 1e-3, 1e-1, log=True),
        # "max_steps": 1e6,
        "lr": trial.suggest_categorical("learning_rate", lr_values),
        "gamma": trial.suggest_float("gamma", 0.9, 0.99, log=True),
        "beta": trial.suggest_categorical("beta", beta_values),
        "minibatch_size": 2 ** trial.suggest_int("minibatch_size", 4, 8, log=True),
        "hidden_size": 2 ** trial.suggest_int("hidden_size", 7, 9, log=True),
        "gradient_clipping": bool(trial.suggest_int("gradient_clipping", 0, 1)),
        "max_episodes": 14000,
        "num_agents": 8,
        "device": "cuda",
        "input_shape": (4, 84, 84),
        "n_actions": len(MOVEMENT),
        "model_path": "trained_models",
        "test_episodes": 5,
        "seed": 42,
    }

    env = gym_tetris.make("TetrisA-v3", apply_api_compatibility=True)
    env = JoypadSpace(env, MOVEMENT)
    env = GrayScaleObservation(env)
    env = FrameStack(env, 4)

    global_model = ActorCriticLSTM(
        params["input_shape"], params["n_actions"], params["hidden_size"]
    )
    global_model.share_memory()

    optimizer = SharedAdam(global_model.parameters(), lr=params["lr"])
    optimizer.share_memory()

    processes = []
    global_steps = mp.Value("i", 0)

    progress_process = mp.Process(
        target=update_progress, args=(global_steps, params["max_episodes"], "Total Episodes", "episodes")
    )
    progress_process.start()
    processes.append(progress_process)
    time.sleep(0.1)

    for rank in range(params["num_agents"]):
        p = mp.Process(
            target=a3c_train, args=(rank, params, global_model, optimizer, global_steps)
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for process in processes:
        time.sleep(0.1)
        process.join()

    done = True
    total_reward = 0
    total_lines = 0
    total_blocks = 0
    episode_length = 0
    num_tests = 0

    global_model.eval()

    while num_tests <= params["test_episodes"]:
        with torch.no_grad():
            if done:
                episode_reward = 0
                state, info = env.reset()
                hx = torch.zeros(1, params["hidden_size"])
                cx = torch.zeros(1, params["hidden_size"])
            else:
                hx = hx.data
                cx = cx.data

            state = preprocess_frame_stack(state)
            policy, _, hx, cx = global_model(state.unsqueeze(0), hx, cx)
            probs = F.softmax(policy, dim=1)
            action = probs.cpu().numpy().argmax()

        state, reward, done, _, info = env.step(action)
        episode_reward += reward
        episode_length += 1

        if done:
            num_tests += 1
            total_reward += episode_reward
            total_lines += info["number_of_lines"]
            total_blocks += sum(info["statistics"].values())
            episode_reward = 0
    
    save_model(global_model, optimizer, trial.number, params)

    env.close()
    mean_reward = total_reward / num_tests
    mean_lines = total_lines / num_tests
    mean_blocks = total_blocks / num_tests
    mean_eps_length = episode_length / num_tests

    values = (mean_reward, mean_blocks, mean_lines, mean_eps_length)

    return values


if __name__ == "__main__":
    try:
        optuna.logging.get_logger("optuna").addHandler(
            logging.StreamHandler(sys.stdout)
        )
        storage = optuna.storages.RDBStorage(
            url="sqlite:///a3c_tetris_hpo.db",
            engine_kwargs={"connect_args": {"timeout": 30}},
        )
        study = optuna.create_study(
            study_name="a3c-tetris",
            storage=storage,
            load_if_exists=True,
            directions=["maximize", "maximize", "maximize", "maximize"],
        )
        n_trials = 60 - len(study.trials)

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print("Number of finished trials: ", len(study.trials))

        print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

        for index, metric_name in metrics:
            print_best_trial(study.best_trials, index, metric_name)

        print("Tuning selesai.")

    except (KeyboardInterrupt, optuna.exceptions.OptunaError) as e:
        print(f"Error: {e}")
        print("Tuning berhenti.")

    finally:
        print("Proses tuning dihentikan")
