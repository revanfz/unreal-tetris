import custom_env

import os
import torch
import timeit
import optuna
import gymnasium as gym
import torch.nn.functional as F
import torch.multiprocessing as mp

from tqdm import tqdm 
from model import ActorCriticLSTM
from optimizer import SharedAdam
from torch.distributions import Categorical
from multiprocessing.sharedctypes import Synchronized


def a3c_train(
    rank: int,
    params: dict,
    global_model: ActorCriticLSTM,
    optimizer: SharedAdam,
    global_episodes: Synchronized,
):
    if rank == 0:
        start_time = timeit.default_timer()

    torch.manual_seed(42 + rank)
    device = params["device"]
    local_model = ActorCriticLSTM(params["input_shape"], params["n_actions"]).to(device)
    local_model.load_state_dict(global_model.state_dict())

    render_mode = "human" if rank == 0 else "rgb_array"
    env = gym.make("SmartTetris-v0", render_mode=render_mode)
    state, info = env.reset()
    done = False

    while global_episodes.value <= params["max_episodes"]:
        optimizer.zero_grad()
        local_model.load_state_dict(global_model.state_dict())
        
        with global_episodes.get_lock():
            global_episodes.value += 1

        hx = torch.zeros((1, 256), dtype=torch.float, device=device)
        cx = torch.zeros((1, 256), dtype=torch.float, device=device)

        values, log_probs, rewards, entropies, masks = [], [], [], [], []

        episode_reward = 0

        for _ in range(params["sync_steps"]):
            matrix_image = torch.from_numpy(state["matrix_image"]).to(device)
            falling_shape = torch.from_numpy(state["falling_shape"]).to(device)
            state = torch.cat((matrix_image, falling_shape), dim=0).to(device)

            policy, value, hx, cx = local_model(state, hx, cx)

            probs = F.softmax(policy, dim=1)
            dist = Categorical(probs)
            action = dist.sample()

            state, reward, done, _, info = env.step(action.item())

            episode_reward += reward

            values.append(value)
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            entropies.append(dist.entropy())
            masks.append(1-done)

            if done:
                state, info = env.reset()
                break

        R = values[-1] * masks[-1]
        gae = torch.zeros(1, 1).to(device)
        actor_loss = 0
        critic_loss = 0
        values.append(R)

        for t in reversed(range(len(rewards))):
            R = rewards[t] + params["gamma"] * R * masks[t]
            advantage = R - values[t]
            critic_loss += advantage.pow(2).mean()

            # GAE
            delta_t = rewards[t] + params["gamma"] * values[t + 1] * masks[t] - values[t]
            gae = gae * params["gamma"] * params["tau"] * masks[t] + delta_t

            actor_loss -= log_probs[t] * gae.detach() - params["beta"] * entropies[t]

        total_loss = actor_loss + 0.5 * critic_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), params["max_grad_norm"])
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad
        optimizer.step()

        print(f"Agent {rank} finished. Episode {global_episodes.value}, Total Reward: {episode_reward}")

    if rank == 0:
        end_time = timeit.default_timer()
        print("The code runs for %.2f s " % (end_time - start_time))
    env.close()


def objective(trial: optuna.Trial):
    params = {
        "model_path": "trained_models",
        "num_agents": trial.suggest_int("num_agents", 4, 12),
        "input_shape": (2, 84, 84),
        "n_actions": 36,
        "lr": trial.suggest_float("lr", 1e-5, 1e-3),
        "gamma": trial.suggest_float("gamma", 0.95, 0.99),
        "sync_steps": trial.suggest_int("sync_steps", 5, 20),
        "beta": trial.suggest_float("beta", 1e-4, 1e-2),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 1.0),
        "max_episodes": 10,
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "tau": 0.95
    }

    global_model = ActorCriticLSTM(params["input_shape"], params["n_actions"]).to(
        params["device"]
    )
    global_model.share_memory()
    model_file = "{}/a3c_tetris_lstm.pt".format(params["model_path"])
    if os.path.isfile(model_file):
        global_model.load_state_dict(torch.load(model_file))

    optimizer = SharedAdam(global_model.parameters(), lr=params["lr"])
    optimizer.share_memory()

    processes = []
    global_episodes = mp.Value("i", 0)
    for rank in range(params["num_agents"]):
        p = mp.Process(
            target=a3c_train,
            args=(rank, params, global_model, optimizer, global_episodes),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    env = gym.make("SmartTetris-v0", render_mode="human")
    state, info = env.reset()
    done = False
    total_reward = 0
    hx = torch.zeros((1, 256), dtype=torch.float, device=params["device"])
    cx = torch.zeros((1, 256), dtype=torch.float, device=params["device"])

    while not done:
        matrix_image = torch.from_numpy(state["matrix_image"]).to(params["device"])
        falling_shape = torch.from_numpy(state["falling_shape"]).to(params["device"])
        state = torch.cat((matrix_image, falling_shape), dim=0).to(params["device"])
        policy, _, hx, cx = global_model(state, hx, cx)
        action = torch.argmax(policy).item()
        state, reward, done, _, info = env.step(action)
        total_reward += reward

    env.close()

    return total_reward


if __name__ == '__main__':
    try:
        storage = optuna.storages.RDBStorage(
            url="sqlite:///a3c-lstm_tetris_hpo.db",
            engine_kwargs={"connect_args": {"timeout": 30}}
        )
        study = optuna.create_study(
            study_name="a3c-lstm_tetris",
            storage=storage,
            direction='maximize',
            load_if_exists=True
        )
        study.optimize(objective, n_trials=50)

        print("Best trial:")
        trial = study.best_trial
        print("Value: ", trial.value)
        print("Params: ")
        for key, value in trial.params.items():
            print("     {}: {} ".format(key, value))
    
    except (KeyboardInterrupt, optuna.exceptions.OptunaError) as e:
        print(f"Program dihentikan {e}...")