import os
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import timeit
import optuna
import gym_tetris
import torch.nn.functional as F
import torch.multiprocessing as mp

from model import ActorCriticFF, ActorCriticLSTM
from optimizer import SharedAdam
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
from multiprocessing.sharedctypes import Synchronized
from gym.wrappers import FrameStack, GrayScaleObservation
from utils import ensure_share_grads, preprocess_frame_stack, preprocessing


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
    global_steps: Synchronized,
) -> None:
    try:
        device = opt["device"]

        torch.manual_seed(42 + rank)
        env = gym_tetris.make("TetrisA-v3", apply_api_compatibility=True)
        env = JoypadSpace(env, MOVEMENT)
        env = GrayScaleObservation(env)
        env = FrameStack(env, 4)

        local_model = ActorCriticLSTM(opt["input_shape"], opt["n_actions"]).to(device)
        local_model.train()

        done = True

        while global_steps.value <= opt["max_steps"]:
            optimizer.zero_grad()
            local_model.load_state_dict(global_model.state_dict())

            if done:
                state, info = env.reset()
                hx = torch.zeros(1, 256).to(device)
                cx = torch.zeros(1, 256).to(device)
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

                with global_steps.get_lock():
                    global_steps.value += 1

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
                delta_t = rewards[t] + opt["gamma"] * values[t + 1].data - values[t].data
                gae = gae * opt["gamma"] + delta_t
                actor_loss -= (log_probs[t] * gae) - (opt["beta"] * entropies[t])

            total_loss = actor_loss + 0.5 * critic_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40)
            ensure_share_grads(
                local_model=local_model, global_model=global_model, device=device
            )
            optimizer.step()

            with global_steps.get_lock():
                global_steps.value += 1

            print(
                f"Agent {rank} finished episode {global_steps.value}"
            )

        print(f"Agent {rank} training process finished.")

    except (KeyboardInterrupt, torch.multiprocessing.ProcessError) as e:
        print(e)

    finally:
        env.close()


def objective(trial: optuna.Trial):
    params = {
        "num_agents": 4,
        "lr": trial.suggest_float("lr", 1e-8, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999, log=True),
        "minibatch_size": trial.suggest_int("minibatch_size", 5, 75, log=True),
        "beta": trial.suggest_float("beta", 1e-2, 5e-1, log=True),
        "device": "cpu",
        "input_shape": (4, 84, 84),
        "n_actions": len(MOVEMENT),
        "model_path": "/content/drive/MyDrive/TA/trained_models",
        "max_steps": 1e5,
        "test_episodes": 10,
        "seed": 42,
    }

    env = gym_tetris.make("TetrisA-v3", apply_api_compatibility=True)
    env = JoypadSpace(env, MOVEMENT)
    env = GrayScaleObservation(env)
    env = FrameStack(env, 4)

    global_model = ActorCriticLSTM(params["input_shape"], params["n_actions"])
    global_model.share_memory()

    optimizer = SharedAdam(global_model.parameters(), lr=params["lr"])
    optimizer.share_memory()

    processes = []
    global_steps = mp.Value("i", 0)

    for rank in range(params["num_agents"]):
        p = mp.Process(
            target=a3c_train,
            args=(rank, params, global_model, optimizer, global_steps)
        )
        p.start()
        processes.append(p)

    for process in processes:
        process.join()

    done = True
    total_reward = 0
    total_lines = 0
    total_blocks = 0
    episode_length = 0
    num_tests = 0

    global_model.eval()

    while num_tests < params["test_episodes"]:
        with torch.no_grad():
            if done:
                episode_reward = 0
                state, info = env.reset()
                hx = torch.zeros(1, 256)
                cx = torch.zeros(1, 256)
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

    env.close()
    mean_reward = total_reward / num_tests
    mean_lines = total_lines / num_tests
    mean_blocks = total_blocks / num_tests
    mean_eps_length = episode_length / num_tests

    return mean_reward, mean_blocks, mean_lines, mean_eps_length


if __name__ == "__main__":
    try:
        storage = optuna.storages.RDBStorage(
            url="sqlite:///test_hpo.db",
            engine_kwargs={"connect_args": {"timeout": 30}}
        )
        study = optuna.create_study(
            study_name="a3c-tetris",
            storage=storage,
            directions=['maximize', 'maximize', 'maximize']
        )
        study.optimize(objective, n_trials=100)

        print("Number of finished trials: ", len(study.trials))

        print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

        metrics = [
            (0, "reward"),
            (1, "blocks placed"),
            (2, "lines cleared")
        ]

        for index, metric_name in metrics:
            print_best_trial(study.best_trials, index, metric_name)

        print("Tuning selesai.")

    except (KeyboardInterrupt, optuna.exceptions.OptunaError) as e:
        print(f"Error: {e}")
        print("Tuning berhenti.")

    finally:
        print("Proses tuning dihentikan")