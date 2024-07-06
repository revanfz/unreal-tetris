import torch
import timeit
import gymnasium as gym
import torch.nn.functional as F

import custom_env

from optimizer import SharedAdam
from torch.distributions import Categorical
from model import ActorCriticFF, ActorCriticLSTM
from torch.utils.tensorboard import SummaryWriter
from multiprocessing.sharedctypes import Synchronized


def local_train_ff(
    rank: int,
    opt: dict,
    global_model: ActorCriticFF,
    optimizer: SharedAdam,
    global_episodes: Synchronized,
    global_steps: Synchronized,
    res_queue: torch.multiprocessing.Queue,
):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if rank == 0:
            start_time = timeit.default_timer()

        torch.manual_seed(42 + rank)
        writer = SummaryWriter(opt.log_path, max_queue=opt.num_agents)

        opt.render_mode = opt.render_mode if rank != 0 else "human"
        env = gym.make("SmartTetris-v0", render_mode=opt.render_mode)

        local_model = ActorCriticFF((2, 84, 84), env.action_space.n).to(device)
        local_model.train()

        state, info = env.reset()

        done = False
        curr_episode = 0
        num_games = 0

        while global_steps.value <= opt.max_steps:
            optimizer.zero_grad()
            local_model.load_state_dict(global_model.state_dict())

            if rank == 0:
                if (
                    global_steps.value % opt.save_interval == 0
                    and global_steps.value > 0
                ):
                    torch.save(
                        global_model.state_dict(),
                        "{}/a3c_tetris_ff.pt".format(opt.model_path),
                    )

            episode_rewards = 0
            values, log_probs, rewards, entropies, masks = [], [], [], [], []

            for step in range(opt.sync_steps):
                matrix_image = torch.from_numpy(state["matrix_image"]).to(device)
                falling_shape = torch.from_numpy(state["falling_shape"]).to(device)
                state = torch.cat((matrix_image, falling_shape), dim=0).to(device)

                policy, value = local_model(state)

                probs = F.softmax(policy, dim=1)
                m = Categorical(probs=probs)
                action = m.sample()
                log_prob = m.log_prob(action)
                entropy = m.entropy()

                state, reward, done, _, info = env.step(action.item())
                episode_rewards += reward

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)
                masks.append(1-done)

                with global_steps.get_lock():
                    global_steps.value += 1

                if done:
                    num_games += 1
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
                    rewards[t] + opt.gamma * masks[t] * values[t + 1] - values[t]
                )
                gae = td_error + opt.gamma * opt.tau * masks[t] * gae
                advantages[t] = gae

            critic_loss = advantages.pow(2).mean()
            actor_loss = (
                -(advantages.detach() * log_probs).mean() - opt.beta * entropies.mean()
            )
            total_loss = actor_loss + 0.5 * critic_loss

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

            mean_reward = episode_rewards / len(rewards)
            res_queue.put({
                "episode": global_episodes.value,
                "mean_reward": mean_reward
            })

            writer.add_scalar("Agent {}/Loss".format(rank), total_loss, curr_episode)
            writer.add_scalar(
                "Agent {}/Block Placed".format(rank),
                info["block_placed"],
                num_games
            )
            writer.add_scalar(
                "Agent {}/Lines Cleared ".format(rank),
                info["total_lines"],
                num_games
            )

            if done:
                state, info = env.reset()

            print(
                "Process {} finished episode {}. Mean reward: {}".format(
                    rank, global_episodes.value, mean_reward
                )
            )

        print("Training process {} terminated".format(rank))
        if rank == 0:
            end_time = timeit.default_timer()
            print("The code runs for %.2f s " % (end_time - start_time))

    except (KeyboardInterrupt, torch.multiprocessing.ProcessError) as e:
        print(f"Training dihentikan... {e}")

    finally:
        env.close()
        writer.close()
        res_queue.put(None)


def local_train_lstm(
    rank: int,
    opt: dict,
    global_model: ActorCriticLSTM,
    optimizer: SharedAdam,
    global_episodes: Synchronized,
    global_steps: Synchronized,
    episode_reward: torch.multiprocessing.Queue,
    step_reward: torch.multiprocessing.Queue,
    loss_model: torch.multiprocessing.Queue,
):
    try:
        if rank == 0:
            start_time = timeit.default_timer()

        torch.manual_seed(42 + rank)
        writer = SummaryWriter(opt.log_path, max_queue=opt.num_agents)

        opt.render_mode = opt.render_mode if rank != 0 else "human"
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
                matrix_image = torch.from_numpy(state["matrix_image"]).to(
                    local_model.device
                )
                falling_shape = torch.from_numpy(state["falling_shape"]).to(
                    local_model.device
                )
                state = torch.cat((matrix_image, falling_shape), dim=0).to(
                    local_model.device
                )

            curr_eps += 1
            with global_episodes.get_lock():
                global_episodes.value += 1

            optimizer.zero_grad()
            local_model.load_state_dict(global_model.state_dict())

            if rank == 0:
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
                matrix_image = torch.from_numpy(state["matrix_image"]).to(
                    local_model.device
                )
                falling_shape = torch.from_numpy(state["falling_shape"]).to(
                    local_model.device
                )
                state = torch.cat((matrix_image, falling_shape), dim=0).to(
                    local_model.device
                )

                curr_step += 1
                with global_steps.get_lock():
                    global_steps.value += 1
                    step_reward.put(reward)

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)

                if done:
                    state, info = env.reset()
                    matrix_image = torch.from_numpy(state["matrix_image"]).to(
                        local_model.device
                    )
                    falling_shape = torch.from_numpy(state["falling_shape"]).to(
                        local_model.device
                    )
                    state = torch.cat((matrix_image, falling_shape), dim=0).to(
                        local_model.device
                    )
                    break

            R = values[-1] * int(not done)

            actor_loss = 0
            critic_loss = 0
            entropy_loss = 0

            for t in reversed(range(len(rewards))):
                R = rewards[t] + opt.gamma * R
                advantage = R - values[t]
                actor_loss += log_probs[t] * advantage.detach()
                critic_loss += advantage**2 / 2
                entropy_loss += entropies[t]

            total_loss = -actor_loss + 0.5 * critic_loss - opt.beta * entropy

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 40)

            for local_param, global_param in zip(
                local_model.parameters(), global_model.parameters()
            ):
                if global_param.grad is not None:
                    break
                global_param._grad = local_param.grad

            optimizer.step()

            mean_eps_r = sum(rewards) / len(rewards)
            episode_reward.put(mean_eps_r)
            loss_model.put(total_loss.detach().cpu().numpy()[0])

            print(
                "Agent {} finished training: episode {}".format(
                    rank, global_episodes.value
                )
            )

            if rank == 0:
                writer.add_scalar(
                    "Model Loss",
                    total_loss,
                    global_episodes.value,
                )

            writer.add_scalar(
                "Avg reward",
                mean_eps_r,
                global_episodes.value,
            )

            writer.add_scalar("Agent {}/Loss".format(rank), total_loss, curr_eps)

        print("Training process of agent {} terminated.".format(rank))
        if rank == 0:
            end_time = timeit.default_timer()
            print("The code runs for %.2f s " % (end_time - start_time))

    except (KeyboardInterrupt, torch.multiprocessing.ProcessError) as e:
        print(f"Training {rank} dihentikan {e}")
        raise KeyboardInterrupt

    finally:
        writer.close()
        episode_reward.put(None)
        step_reward.put(None)
        loss_model.put(None)
