from time import sleep
import torch
import timeit
import gymnasium as gym
import torch.nn.functional as F

import custom_env

from model import ActorCriticFF, ActorCriticLSTM
from optimizer import SharedAdam
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from multiprocessing.sharedctypes import Synchronized

def local_train_ff(
    index: int,
    opt: tuple,
    global_model: ActorCriticFF,
    optimizer: SharedAdam,
    global_episodes: Synchronized,
    global_steps: Synchronized,
    global_rewards: Synchronized,
    res_queue: torch.multiprocessing.Queue,
    timestamp=False,
):
    try:
        if timestamp:
            start_time = timeit.default_timer()

        torch.manual_seed(42 + index)
        opt.render_mode = opt.render_mode if index != 0 else "human"
        writer = SummaryWriter(opt.log_path, max_queue=opt.num_agents)
        env = gym.make("SmartTetris-v0", render_mode=opt.render_mode)
        local_model = ActorCriticFF(opt.framestack, env.action_space.n)
        if local_model.device == "cuda":
            local_model.cuda()
        local_model.train()

        obs, info = env.reset()
        obs = torch.from_numpy(obs["matrix_image"]).to(local_model.device)
        state = torch.zeros((opt.framestack, 84, 84), device=local_model.device)
        state[-1] = obs
        done = True
        curr_episode = 0

        while global_steps.value <= opt.max_steps:
            optimizer.zero_grad()
            local_model.load_state_dict(global_model.state_dict())

            if timestamp:
                if (
                    global_steps.value % opt.save_interval == 0
                    and global_steps.value > 0
                ):
                    torch.save(
                        global_model.state_dict(),
                        "{}/a3c_tetris.pt".format(opt.model_path),
                    )

            log_probs = torch.zeros(opt.sync_steps, device=local_model.device)
            values = torch.zeros(opt.sync_steps, device=local_model.device)
            rewards = torch.zeros(opt.sync_steps, device=local_model.device)
            entropies = torch.zeros(opt.sync_steps, device=local_model.device)

            for step in range(opt.sync_steps):
                policy, value = local_model(state)
                probs = F.softmax(policy, dim=1)

                m = Categorical(probs=probs)
                action = m.sample()
                log_prob = m.log_prob(action)
                entropy = m.entropy()

                obs, reward, done, _, info = env.step(action)
                obs = torch.from_numpy(obs["matrix_image"]).to(local_model.device)
                state = torch.cat((state[1:], obs), dim=0).to(local_model.device)

                values[step] = value
                log_probs[step] = log_prob
                rewards[step] = reward
                entropies[step] = entropy

                with global_steps.get_lock():
                    global_steps.value += 1

                if done:
                    values = values[: step + 1]
                    log_probs = log_probs[: step + 1]
                    rewards = rewards[: step + 1]
                    entropies = entropies[: step + 1]

                    obs, info = env.reset()
                    obs = torch.from_numpy(obs["matrix_image"])
                    state = torch.zeros(
                        (opt.framestack, 84, 84), device=local_model.device
                    )
                    state[-1] = obs
                    break

            total_loss = local_model.calculate_loss(
                done, values, log_probs, entropies, rewards, opt.gamma, opt.beta
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40)

            for local_param, global_param in zip(
                local_model.parameters(), global_model.parameters()
            ):
                if global_param.grad is not None:
                    break
                global_param._grad = local_param.grad

            optimizer.step()

            curr_episode += 1

            with global_episodes.get_lock():
                global_episodes.value += 1

            with global_rewards.get_lock():
                eps_r = rewards.sum().cpu().numpy()
                global_rewards.value = global_rewards.value + eps_r

                mean_reward = global_rewards.value / global_episodes.value
            res_queue.put(mean_reward)

            writer.add_scalar("Agent {}/Loss".format(index), total_loss, curr_episode)
            writer.add_scalar(
                "Block Placed Agent {}".format(index),
                info["block_placed"],
                curr_episode,
            )
            writer.add_scalar(
                "Lines Cleared Agent {}".format(index),
                info["total_lines"],
                curr_episode,
            )
            writer.add_scalar(
                "Global Rewards",
                mean_reward,
                global_episodes.value,
            )

            print(
                "Process {} Finished episode {}. Mean reward: {}".format(
                    index, global_episodes.value, mean_reward
                )
            )

        print("Training process {} terminated".format(index))
        if timestamp:
            end_time = timeit.default_timer()
            print("The code runs for %.2f s " % (end_time - start_time))
    except (KeyboardInterrupt, torch.multiprocessing.ProcessError) as e:
        print(f"Training dihentikan... {e}")
    finally:
        writer.close()
        res_queue.put(None)



def local_train_lstm(
    index: int,
    opt: tuple,
    global_model: ActorCriticLSTM,
    optimizer: SharedAdam,
    global_episodes: Synchronized,
    global_steps: Synchronized,
    res_queue: torch.multiprocessing.Queue,
):
    try:
        if index == 0:
            start_time = timeit.default_timer()

        torch.manual_seed(42 + index)
        writer = SummaryWriter(opt.log_path, max_queue=opt.num_agents)

        opt.render_mode = opt.render_mode if index != 0 else "human"
        env = gym.make("SmartTetris-v0", render_mode=opt.render_mode)

        local_model = ActorCriticLSTM(2, env.action_space.n)
        if local_model.device == "cuda":
            local_model.cuda()

        done = True
        curr_eps = 0
        curr_step = 0

        while global_steps.value <= opt.max_steps:
            if done:
                state, info = env.reset()
                matrix_image = torch.from_numpy(state['matrix_image']).to(local_model.device)
                falling_shape = torch.from_numpy(state['falling_shape']).to(local_model.device)
                state = torch.cat((matrix_image, falling_shape), dim=0).to(local_model.device)

            curr_eps += 1
            with global_episodes.get_lock():
                global_episodes.value += 1

            optimizer.zero_grad()
            local_model.load_state_dict(global_model.state_dict())

            if index == 0:
                if (
                    global_steps.value % opt.save_interval == 0
                    and global_steps.value > 0
                ):
                    torch.save(
                        global_model.state_dict(),
                        "{}/a3c_tetris_lstm.pt".format(opt.model_path),
                    )

            if done:
                hx = torch.zeros((1, 256), dtype=torch.float)
                cx = torch.zeros((1, 256), dtype=torch.float)
            else:
                hx = hx.detach()
                cx = cx.detach()
            if local_model.device == "cuda":
                hx = hx.cuda()
                cx = cx.cuda()

            log_probs = []
            values = []
            rewards = []
            entropies = []

            for step in range(opt.sync_steps):
                policy, value, hx, cx = local_model(state, hx, cx)
                probs = F.softmax(policy, dim=1)

                distribution = Categorical(probs=probs)
                action = distribution.sample()
                log_prob = distribution.log_prob(action)
                entropy = distribution.entropy()

                state, reward, done, _, info = env.step(action)
                sleep(0.25)
                matrix_image = torch.from_numpy(state['matrix_image']).to(local_model.device)
                falling_shape = torch.from_numpy(state['falling_shape']).to(local_model.device)
                state = torch.cat((matrix_image, falling_shape), dim=0).to(local_model.device)

                curr_step += 1
                with global_steps.get_lock():
                    global_steps.value += 1
                    
                    writer.add_scalar(
                        "Global reward/step",
                        global_steps.value,
                        reward
                    )
                
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)

                if done:
                    state, info = env.reset()
                    matrix_image = torch.from_numpy(state['matrix_image']).to(local_model.device)
                    falling_shape = torch.from_numpy(state['falling_shape']).to(local_model.device)
                    state = torch.cat((matrix_image, falling_shape), dim=0).to(local_model.device)
                    break
            
            R = values[-1] * int(not done)

            actor_loss = 0
            critic_loss = 0
            entropy_loss = 0

            for t in reversed(range(len(rewards))):
                R = rewards[t] + opt.gamma * R
                advantage = R - values[t]
                actor_loss += log_probs[t] * advantage
                critic_loss += advantage.pow(2).mean()
                entropy_loss += entropies[t]

            total_loss = -actor_loss + 0.5 * critic_loss - opt.beta * entropy

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40)
            
            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                if global_param.grad is not None:
                    break
                global_param._grad = local_param.grad

            optimizer.step()

            mean_eps_r = sum(rewards) / len(rewards)
            res_queue.put(mean_eps_r)

            writer.add_scalar(
                "Model Loss",
                global_steps.value,
                total_loss
            )

            writer.add_scalar(
                "Avg reward",
                global_steps.value,
                mean_eps_r
            )

        print("Training process {} terminated".format(index))
        if index == 0:
            end_time = timeit.default_timer()
            print("The code runs for %.2f s " % (end_time - start_time))

    except (KeyboardInterrupt, torch.multiprocessing.ProcessError) as e:
        print(f"Training {index} dihentikan {e}")
        raise KeyboardInterrupt

    finally:
        writer.close()
        res_queue.put(None)
