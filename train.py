from multiprocessing.managers import SyncManager
from multiprocessing.synchronize import Event
import time
import torch
import timeit
import gym_tetris
import torch.nn.functional as F

from logging import Logger
from optimizer import SharedAdam
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
from torch.distributions import Categorical
from model import ActorCriticFF, ActorCriticLSTM
from torch.utils.tensorboard import SummaryWriter
from multiprocessing.sharedctypes import Synchronized
from gym.wrappers import FrameStack, GrayScaleObservation, FrameStack
from utils import ensure_share_grads, preprocess_frame_stack, preprocessing


def train_model(
    rank: int,
    opt: dict,
    global_model: ActorCriticFF,
    optimizer: SharedAdam,
    global_episodes: Synchronized,
    global_steps: Synchronized,
    global_rewards: Synchronized,
    res_queue: torch.multiprocessing.Queue,
) -> None:
    try:
        if not rank:
            start_time = timeit.default_timer()

        device = "cuda" if opt.use_cuda else "cpu"
        writer = SummaryWriter(opt.log_path)

        torch.manual_seed(42 + rank)
        env = gym_tetris.make("TetrisA-v3", apply_api_compatibility=True)
        env = JoypadSpace(env, MOVEMENT)
        env = GrayScaleObservation(env)
        env = FrameStack(env, 4)

        local_model = ActorCriticFF((opt.framestack, 84, 84), env.action_space.n).to(
            device
        )
        local_model.train()

        done = True

        curr_episodes = 0

        while global_steps.value <= opt.max_steps:
            if done:
                state, info = env.reset()

            optimizer.zero_grad()
            local_model.load_state_dict(global_model.state_dict())

            # if rank == 0:
            #     if curr_episodes % opt.save_interval == 0:
            #         torch.save(
            #             global_model.state_dict,
            #             "{}/a3c_tetris_ff.pt".format(opt.model_path),
            #         )

            episode_rewards = 0
            values, log_probs, rewards, entropies = [], [], [], []

            for step in range(opt.minibatch_size):
                state = preprocess_frame_stack(state).to(device)
                policy, value = local_model(state)

                probs = F.softmax(policy, dim=1)
                dist = Categorical(probs=probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                state, reward, done, _, info = env.step(action.item())
                if not rank:
                    env.render()
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
                _, value = local_model(bootstrap_state)
                R = value.detach()
            values.append(R)

            actor_loss = 0
            critic_loss = 0

            for t in reversed(range(len(rewards))):
                R = opt.gamma * R + rewards[t]
                advantage = R - values[t]
                critic_loss += advantage.pow(2).mean()

                # GAE
                delta_t = rewards[t] + opt.gamma * values[t + 1] - values[t]
                gae = gae * opt.gamma  + delta_t
                actor_loss -= (log_probs[t] * gae) - (opt.beta * entropies[t])

            total_loss = actor_loss + 0.5 * critic_loss
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
            ensure_share_grads(
                local_model=local_model, global_model=global_model, device=device
            )
            optimizer.step()

            curr_episodes += 1
            with global_episodes.get_lock():
                global_episodes.value += 1

            with global_rewards.get_lock():
                global_rewards.value += episode_rewards
                mean_reward = global_rewards.value / global_episodes.value

            res_queue.put(
                {
                    "episode": global_episodes.value,
                    "mean_reward": mean_reward,
                    "entropy": (sum(entropies) / len(entropies)).detach().item(),
                }
            )

            writer.add_scalar(f"Agent {rank}/Losses", total_loss, curr_episodes)
            writer.add_scalar(f"Agent {rank}/Rewards", episode_rewards, curr_episodes)
            writer.add_scalar(
                f"Agent {rank}/Lines cleared", info["number_of_lines"], curr_episodes
            )
            writer.add_scalar(
                f"Agent {rank}/Block Placed",
                sum(info["statistics"].values()),
                curr_episodes,
            )

            print(
                f"Agent {rank} finished episode {global_episodes.value},\
                reward: {mean_reward}"
            )

        print(f"Agent {rank} training process terminated.")
       
        if not rank:
            end_time = timeit.default_timer()
            print("The code runs for %.2f s " % (end_time - start_time))

    except (KeyboardInterrupt, torch.multiprocessing.ProcessError) as e:
        print(e)

    finally:
        env.close()
        writer.close()
        res_queue.put(None)


def train_model_lstm(
    rank: int,
    opt: dict,
    global_model: ActorCriticFF,
    optimizer: SharedAdam,
    global_episodes: Synchronized,
    global_steps: Synchronized,
    global_rewards: Synchronized,
    stop_event: Event,
    start_train_time: int,
    max_train_time: int,
    res_queue: torch.multiprocessing.Queue,
    shared_dict: SyncManager,
    initial_eps: int = 0,
) -> None:
    try:
        if not rank:
            start_time = timeit.default_timer()

        device = "cuda" if opt.use_cuda else "cpu"
        writer = SummaryWriter(opt.log_path)

        torch.manual_seed(42 + rank)

        env = gym_tetris.make("TetrisA-v3", apply_api_compatibility=True)
        env = JoypadSpace(env, MOVEMENT)
        env = GrayScaleObservation(env)
        env = FrameStack(env, 4)

        local_model = ActorCriticLSTM((4, 84, 84), env.action_space.n, opt.hidden_size).to(device)
        local_model.train()

        done = True

        curr_episodes = initial_eps

        while global_steps.value <= opt.max_steps:
            optimizer.zero_grad() # Reset gradient
            local_model.load_state_dict(global_model.state_dict()) # Menyalin nlai parameter global

            # Mereset lingkungan simulasi
            if done:
                state, info = env.reset()
                hx = torch.zeros(1, opt.hidden_size).to(device)
                cx = torch.zeros(1, opt.hidden_size).to(device)
            else:
                hx = hx.data
                cx = cx.data

            episode_rewards = 0
            values, log_probs, rewards, entropies = [], [], [], []

            for step in range(opt.minibatch_size):
                state = preprocess_frame_stack(state).to(device)
                policy, value, hx, cx = local_model(state.unsqueeze(0), hx, cx)

                probs = F.softmax(policy, dim=1)
                log_prob = F.log_softmax(policy, dim=1)
                entropy = -(log_prob * probs).sum(1)
                action = probs.multinomial(1).data
                log_prob = log_prob.gather(1, action)

                state, reward, done, _, info = env.step(action.item())
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
                R = opt.gamma * R + rewards[t]
                advantage = R - values[t]
                critic_loss += 0.5 * advantage.pow(2)

                # GAE
                delta_t = rewards[t] + opt.gamma * values[t + 1].data - values[t].data
                gae = gae * opt.gamma + delta_t
                actor_loss -= (log_probs[t] * gae) - (opt.beta * entropies[t])

            total_loss = actor_loss + 0.5 * critic_loss
            total_loss.backward()

            # torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
            ensure_share_grads(
                local_model=local_model, global_model=global_model, device=device
            )
            optimizer.step()

            curr_episodes += 1
            shared_dict[f"agent_{rank}"] = curr_episodes

            with global_episodes.get_lock():
                global_episodes.value += 1

            with global_rewards.get_lock():
                global_rewards.value += episode_rewards
                mean_reward = global_rewards.value / global_episodes.value

            res_queue.put(
                {
                    "episode": global_episodes.value,
                    "steps": global_steps.value,
                    "mean_reward": mean_reward,
                    "entropy": (sum(entropies) / len(entropies)).detach().item(),
                }
            )

            writer.add_scalar(f"Agent {rank}/Losses", total_loss, curr_episodes)
            writer.add_scalar(f"Agent {rank}/Rewards", episode_rewards, curr_episodes)
            writer.add_scalar(
                f"Agent {rank}/Lines cleared", info["number_of_lines"], curr_episodes
            )
            writer.add_scalar(
                f"Agent {rank}/Block Placed",
                sum(info["statistics"].values()),
                curr_episodes,
            )

            print(
                f"Agent {rank} finished episode {global_episodes.value},\
                reward: {mean_reward}"
            )

        print(f"Agent {rank} training process terminated.")

        torch.save(
            global_model.state_dict(),
            "{}/baseline_a3c_tetris.pt".format(opt.model_path),
        )

        if not rank:
            end_time = timeit.default_timer()
            print("The code runs for %.2f s " % (end_time - start_time))

    except (KeyboardInterrupt, torch.multiprocessing.ProcessError) as e:
        print(e)

    finally:
        shared_dict[f"agent_{rank}"] = curr_episodes
        env.close()
        writer.close()
        res_queue.put(None)
