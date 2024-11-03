import torch
import torch.nn as nn
import torch.nn.functional as F

from model import UNREAL
from argparse import Namespace
from optimizer import SharedAdam
from replay_buffer import ReplayBuffer
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from multiprocessing.sharedctypes import Synchronized
from utils import ensure_share_grads, make_env, pixel_diff, preprocessing


def worker(
    rank: int,
    global_model: UNREAL,
    optimizer: SharedAdam,
    global_steps: Synchronized,
    global_episodes: Synchronized,
    params: Namespace,
    device: torch.device,
):
    try:
        finished = False
        torch.manual_seed(42 + rank)

        render_mode = "human" if not rank else "rgb_array"
        env = make_env(
            resize=84, render_mode=render_mode, level=19
        )

        device = torch.device("cpu")
        local_model = UNREAL(
            n_inputs=(84, 84, 3),
            n_actions=env.action_space.n,
            hidden_size=256,
            device=device,
            beta=params.beta,
            gamma=params.gamma
        )
        local_model.train()
        
        if not rank:
            total_lines = 0
            total_score = 0
            writer = SummaryWriter(f"{params.log_path}/unreal-tetris")
            writer.add_graph(local_model, (
                torch.zeros(1, 3, 84, 84).to(device),
                F.one_hot(torch.LongTensor([0]), env.action_space.n).to(device),
                torch.zeros(1, 1).to(device),
                (
                    torch.zeros(1, params.hidden_size).to(device),
                    torch.zeros(1, params.hidden_size).to(device)
                ),
            ))
        experience_replay = ReplayBuffer(2000)

        state, info = env.reset()
        state = preprocessing(state)
        prev_action = F.one_hot(torch.LongTensor([0]), env.action_space.n).to(device)
        prev_reward = torch.zeros(1, 1).to(device)

        while not experience_replay._is_full():
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                policy, value, _, _ = local_model(
                    state_tensor, prev_action, prev_reward, None
                )
                probs = F.softmax(policy, dim=1)
                dist = Categorical(probs=probs)
                action = dist.sample().cpu()

            next_state, reward, done, _, info = env.step(action.item())
            next_state = preprocessing(next_state)
            pixel_change = pixel_diff(state, next_state)
            experience_replay.store(state, reward, action.item(), done, pixel_change)
            
            state = next_state
            prev_action = F.one_hot(action, num_classes=env.action_space.n).to(device)
            prev_reward = torch.FloatTensor([[reward]]).to(device)

            if done:
                state, info = env.reset()
                state = preprocessing(state)

        done = True
        current_episodes = 0

        while global_steps.value <= params.max_steps:
            optimizer.zero_grad()
            local_model.load_state_dict(global_model.state_dict())
            log_probs, entropies, rewards, values, dones = [], [], [], [], []

            if done:
                state, info = env.reset()
                state = preprocessing(state)
                action = F.one_hot(torch.LongTensor([0]), num_classes=env.action_space.n).to(device)
                reward = torch.zeros(1, 1).to(device)
                hx = torch.zeros(1, params.hidden_size).to(device)
                cx = torch.zeros(1, params.hidden_size).to(device)
                episode_reward = 0
            else:
                hx = hx.detach()
                cx = cx.detach()

            for _ in range(params.unroll_steps):
                state_tensor =  torch.from_numpy(state).unsqueeze(0).to(device)
                policy, _, hx, cx = local_model(
                    state_tensor, action, reward, (hx, cx)
                )

                probs = F.softmax(policy, dim=1)
                dist = Categorical(probs=probs)
                action = dist.sample().detach()

                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                next_state, reward, done, _, info = env.step(action.item())
                next_state = preprocessing(next_state)
                pixel_change = pixel_diff(state, next_state)
                experience_replay.store(
                    state, reward, action.item(), done, pixel_change
                )
                state = next_state

                episode_reward += reward
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)
                dones.append(done)

                action = F.one_hot(action, num_classes=env.action_space.n).to(
                    device
                )
                reward = torch.FloatTensor([[reward]]).to(device)

                with global_steps.get_lock():
                    global_steps.value += 1

                if done:
                    if not rank:
                        total_lines += info["number_of_lines"]
                        total_score += info["score"]
                        
                        writer.add_scalar(
                            f"Block placed",
                            sum(info["statistics"].values()),
                            global_episodes.value,
                        )
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
                R = R.cpu().detach()

            # Hitung loss A3C
            a3c_loss = local_model.a3c_loss(
                entropies=entropies,
                dones=dones,
                log_probs=log_probs,
                rewards=rewards,
                values=values,
                R=R,
            )

            # Hitung Loss Pixel Control
            # 1.  Sampling replay buffer secara random
            states, rewards, actions, dones, pixel_changes = (
                experience_replay.sample_sequence(params.unroll_steps + 1)
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
                experience_replay.sample_sequence(params.unroll_steps + 1)
            )
            vr_loss = local_model.vr_loss(states, actions, rewards, dones)

            # Penjumlahan loss a3c, pixel control, value replay dan reward prediction
            total_loss = (
                a3c_loss + params.task_weight * pc_loss + rp_loss + vr_loss 
            )

            total_loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), 40)
            ensure_share_grads(
                local_model=local_model, global_model=global_model, device=device
            )
            optimizer.step()

            if not rank:
                writer.add_scalar(f"Losses", total_loss, global_episodes.value)
                writer.add_scalar(f"Rewards", episode_reward, global_episodes.value)
                writer.add_scalar(
                    f"Total lines cleared", total_lines, global_episodes.value
                )
                writer.add_scalar(
                    f"Total scores", total_score, global_episodes.value
                )

                writer.add_scalar(
                    "Entropy", entropy.mean().item(), global_episodes.value
                )

                if current_episodes % 100 == 0:
                    writer.flush()

            current_episodes += 1
            with global_episodes.get_lock():
                global_episodes.value += 1

                if global_episodes.value % params.save_interval == 0:
                    torch.save(
                        {
                            "model_state_dict": global_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "steps": global_steps.value,
                            "episodes": global_episodes.value,
                        },
                        f"{params.model_path}/unreal_checkpoint.tar",
                    )

        if not rank:
            torch.save(global_model.state_dict(), f"{params.model_path}/unreal_tetris.pt")
        finished = True
        print(f"Pelatihan agen {rank} selesai")

    except KeyboardInterrupt as e:
        print(f"Program dihentikan")
        raise KeyboardInterrupt("Program dihentikan")

    except torch.multiprocessing.ProcessError as e:
        print("Unexpected error")
        raise Exception(f"Multiprocessing error\t{e}.")

    except Exception as e:
        print(f"\nError ;X\t{e}")
        raise Exception(f"{e}")

    finally:
        if not finished and not rank:
            torch.save(
                {
                    "model_state_dict": global_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "steps": global_steps.value,
                    "episodes": global_episodes.value,
                },
                f"{params.model_path}/unreal_checkpoint.tar",
            )
        if not rank:
            writer.close()
        env.close()
        print(f"Proses pelatihan agen {rank} dihentikan")
