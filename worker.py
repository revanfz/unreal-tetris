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
from utils import ensure_share_grads, make_env, preprocessing


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
            grayscale=False, framestack=None, resize=84, render_mode=render_mode
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
            writer = SummaryWriter(f"{params.log_path}/a3c-tetris")
            writer.add_graph(local_model, (
                torch.zeros(1, 3, 84, 84).to(device),
                torch.zeros(1, env.action_space.n).to(device),
                torch.zeros(1, 1).to(device),
                (
                    torch.zeros(1, params.hidden_size).to(device),
                    torch.zeros(1, params.hidden_size).to(device)
                ),
            ))
        experience_replay = ReplayBuffer(2000)

        done = True
        current_episodes = 0

        while global_steps.value <= params.max_steps:
            optimizer.zero_grad()
            local_model.load_state_dict(global_model.state_dict())

            for _ in range(params.unroll_steps):
                if done:
                    state, info = env.reset()
                    state = preprocessing(state)
                    prev_action = torch.zeros(1, env.action_space.n).to(device)
                    prev_reward = torch.zeros(1, 1).to(device)
                    hx = torch.zeros(1, params.hidden_size).to(device)
                    cx = torch.zeros(1, params.hidden_size).to(device)
                    episode_reward = 0
                else:
                    hx = hx.data
                    cx = cx.data

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                policy, _, _, hx, cx = local_model(
                    state_tensor, prev_action, prev_reward, (hx, cx)
                )

                dist = Categorical(policy)
                action = dist.sample().detach()

                next_state, reward, done, _, info = env.step(action.item(), info)
                next_state = preprocessing(next_state)

                experience_replay.store(
                    state,
                    prev_action.argmax().item(),
                    prev_reward.item(),
                    next_state,
                    action.item(),
                    reward,
                    done,
                )

                prev_action = F.one_hot(action, num_classes=env.action_space.n).to(
                    device
                )
                prev_reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
                state = next_state

                episode_reward += reward
                with global_steps.get_lock():
                    global_steps.value += 1

            # Hitung loss A3C
            # 1. Sampel replay buffer secara sekuensial
            states, prev_actions, prev_rewards, _, _, _, dones = (
                experience_replay.sample(params.unroll_steps, base=True)
            )
            # 2. Hitung loss actor dan critic
            policy_loss, value_loss, entropy = local_model.a3c_loss(
                states, prev_rewards, prev_actions, dones
            )
            # 3. Jumlahkan loss dengan mengurangi nilai critic loss
            a3c_loss = policy_loss + value_loss

            # Hitung Loss Pixel Control dan Feature Control
            # 1.  Sampling replay buffer secara random
            (
                states,
                prev_actions,
                prev_rewards,
                next_states,
                next_actions,
                next_rewards,
                dones,
            ) = experience_replay.sample(params.unroll_steps)
            # 2a. Hitung loss Pixel Control
            aux_control_loss = local_model.control_loss(
                states,
                prev_actions,
                prev_rewards,
                next_states,
                next_actions,
                next_rewards,
                dones,
            )
            # 2b. Hitung loss Value Replay
            vr_loss = local_model.vr_loss(states, prev_actions, prev_rewards, dones)

            # Hitung Loss Reward Pedictions
            # 1. Sampel 3 frame dengan peluang rewarding state = 0.5
            states, prev_rewards, next_rewards = experience_replay.sample_rp()
            # 2. Hitung loss reward prediction
            rp_loss = local_model.rp_loss(states, prev_rewards, next_rewards[-1])

            # Penjumlahan loss a3c, pixel control, value replay dan reward prediction
            total_loss = (
                a3c_loss + vr_loss + params.task_weight * aux_control_loss + rp_loss
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
                    f"Lines cleared", info["number_of_lines"], global_episodes.value
                )

                writer.add_scalar(
                    f"Block placed",
                    sum(info["statistics"].values()),
                    global_episodes.value,
                )

                writer.add_scalar(
                    "Entropy", entropy.mean().item(), global_episodes.value
                )

                # for name, param in global_model.named_parameters():
                #     writer.add_histogram(f"weights/{name}", param.data.cpu().numpy(), global_episodes.value)
                #     if param.grad is not None:
                #         writer.add_histogram(f"grads/{name}", param.grad.cpu().numpy(), global_episodes.value)

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
                        f"{params.model_path}/a3c_checkpoint.tar",
                    )

        if not rank:
            torch.save(global_model.state_dict(), f"{params.model_path}/a3c_tetris.pt")
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
                f"{params.model_path}/a3c_checkpoint.tar",
            )
        if not rank:
            writer.close()
        env.close()
        print(f"Proses pelatihan agen {rank} dihentikan")
