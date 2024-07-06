import custom_env

import os
import torch
import timeit
import optuna
import gymnasium as gym
import torch.nn.functional as F
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

from model import ActorCriticFF
from optimizer import SharedAdam
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from multiprocessing.sharedctypes import Synchronized

def a3c_train(
    rank: int,
    params: dict,
    global_model: ActorCriticFF,
    optimizer: SharedAdam,
    global_episodes: Synchronized,
):
    if rank == 0:
        start_time = timeit.default_timer()

    torch.manual_seed(42 + rank)
    device = params["device"]
    local_model = ActorCriticFF(params["input_shape"], params["n_actions"]).to(device)
    local_model.train()

    render_mode = "human" if rank == 0 else "rgb_array"
    env = gym.make("SmartTetris-v0", render_mode=render_mode)
    state, info = env.reset()
    done = False

    while global_episodes.value <= params["max_episodes"]:
        optimizer.zero_grad()
        local_model.load_state_dict(global_model.state_dict())

        episode_rewards = 0

        values, log_probs, rewards, entropies, masks = [], [], [], [], []

        for step in range(params["sync_steps"]):
            matrix_image = torch.from_numpy(state["matrix_image"]).to(device)
            falling_shape = torch.from_numpy(state["falling_shape"]).to(device)
            state = torch.cat((matrix_image, falling_shape), dim=0).to(device)

            policy, value = local_model(state)

            probs = F.softmax(policy, dim=1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            state, reward, done, _, info = env.step(action.item())
            episode_rewards += reward

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            masks.append(1-done)

            if done:
                state, info = env.reset()
                break
        
        values = torch.cat(values).to(device)
        log_probs = torch.cat(log_probs).to(device)
        entropies = torch.cat(entropies).to(device)

        T = len(rewards)
        advantages = torch.zeros(T, device=device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + params["gamma"] * masks[t] * values[t + 1] - values[t]
            )
            gae = td_error + params["gamma"] * params["tau"] * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()
        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(advantages.detach() * log_probs).mean() - params["beta"] * entropies.mean()
        )
        total_loss = actor_loss + 0.5 * critic_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), params["max_grad_norm"])
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad
        optimizer.step()
        
        print(f"Agent {rank} finished. episode {global_episodes.value}, reward: {episode_rewards}")

        with global_episodes.get_lock():
            global_episodes.value += 1

    if rank == 0:
        end_time = timeit.default_timer()
        print("The code runs for %.2f s " % (end_time - start_time))
    env.close() 
    

def objective(trial: optuna.Trial):
    params = {
        # "num_agents": trial.suggest_int("num_agents", 4, 12),
        "model_path": "trained_models",
        "num_agents": 10,
        "input_shape": (2, 84, 84),
        "n_actions": 36,
        "lr": trial.suggest_float("lr", 1e-5, 1e-3),
        "gamma": trial.suggest_float("gamma", 0.95, 0.99),
        "sync_steps": trial.suggest_int("sync_steps", 5, 20),
        "beta": trial.suggest_float("beta", 1e-4, 1e-2),
        "max_grad_norm": 40,
        "max_episodes": 10,
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "tau": 0.95
    }

    global_model = ActorCriticFF(params["input_shape"], params["n_actions"]).to(params["device"])
    global_model.share_memory()
    model_file = "{}/a3c_tetris_ff.pt".format(params["model_path"])
    if os.path.isfile(model_file):
        global_model.load_state_dict(torch.load(model_file))

    optimizer = SharedAdam(global_model.parameters(), lr=params["lr"])
    optimizer.share_memory()

    processes = []
    global_steps = mp.Value("i", 0)
    global_episodes = mp.Value("i", 0)

    for rank in range(params["num_agents"]):
        p = mp.Process(
            target=a3c_train,
            args=(rank, params, global_model, optimizer, global_episodes)
        )
        p.start()
        processes.append(p)

    for process in processes:
        process.join()

    env = gym.make("SmartTetris-v0", render_mode="human")
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        matrix_image = torch.from_numpy(state["matrix_image"]).to(params["device"])
        falling_shape = torch.from_numpy(state["falling_shape"]).to(params["device"])
        state = torch.cat((matrix_image, falling_shape), dim=0).to(params["device"])
        
        with torch.no_grad():
            policy, _ = global_model(state)
        action = torch.argmax(policy).item()

        state, reward, done, _, info = env.step(action)
        total_reward += reward

    env.close()

    return total_reward, info["total_lines"]


if __name__ == '__main__':
    try:
        storage = optuna.storages.RDBStorage(
            url="sqlite:///a3c-ff_tetris_hpo.db",
            engine_kwargs={"connect_args": {"timeout": 30}}
        )
        study = optuna.create_study(
            study_name="a3c-ff_tetris",
            storage=storage,
            directions=['maximize', 'maximize'],
            load_if_exists=True
        )
        study.optimize(objective, n_trials=1)

        print("Best trial:")
        for trial in study.best_trials:
            print(f"Trial number: {trial.number}")
            print(f"Params: {trial.params}")
            print(f"Values: {trial.values}")
            print("------------------------")

    
    except (KeyboardInterrupt, optuna.exceptions.OptunaError) as e:
        print(f"Program dihentikan {e}...")