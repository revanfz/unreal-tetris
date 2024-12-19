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
    global_lines: Synchronized,
    params: Namespace,
    device: torch.device,
):
    try:
        level = 19
        torch.manual_seed(42 + rank)

        render_mode = "human" if not rank else "rgb_array"
        env = make_env(
            id="TetrisA-v3",
            resize=84,
            render_mode=render_mode,
            level=level,
            skip=2,
        )
        env.action_space.seed(42 + rank)

        local_model = UNREAL(
            n_inputs=(84, 84, 3),
            n_actions=env.action_space.n,
            hidden_size=params.hidden_size,
            device=device,
            beta=params.beta,
            gamma=params.gamma,
        )
        local_model.load_state_dict(global_model.state_dict())
        local_model.train()

        experience_replay = ReplayBuffer(2000)

        prev_action = F.one_hot(torch.tensor([0]).long(), env.action_space.n).to(device)
        prev_reward = torch.zeros(1, 1, device=device)

        if not rank:
            writer = SummaryWriter(f"{params.log_path}/UNREAL-heuristic")
            with torch.no_grad():
                writer.add_graph(
                    local_model,
                    (
                        torch.zeros(1, 3, 84, 84).to(device),
                        prev_action,
                        prev_reward,
                        (
                            torch.zeros(1, params.hidden_size, device=device),
                            torch.zeros(1, params.hidden_size, device=device),
                        ),
                    ),
                )

        done = True
        while not experience_replay._is_full():
            if done:
                state, info = env.reset(seed=42 + rank)
                state = preprocessing(state)
                hx = torch.zeros(1, params.hidden_size, device=device)
                cx = torch.zeros(1, params.hidden_size, device=device)

            with torch.no_grad():
                state_tensor = torch.tensor(state, device=device).float().unsqueeze(0)
                policy, _, hx, cx = local_model(
                    state_tensor, prev_action, prev_reward, (hx, cx)
                )

                dist = Categorical(probs=policy)
                action = dist.sample().cpu()

            next_state, reward, done, _, info = env.step(action.item())

            next_state = preprocessing(next_state)
            pixel_change = pixel_diff(state, next_state)
            experience_replay.store(state, reward, action.item(), done, pixel_change)

            state = next_state
            prev_action = F.one_hot(action, num_classes=env.action_space.n).to(device)
            prev_reward = torch.tensor([[reward]], device=device).float()

        done = True
        current_episodes = 0

        action = F.one_hot(torch.tensor([0]).long(), env.action_space.n).to(device)
        reward = torch.zeros(1, 1, device=device)

        while global_steps.value <= params.max_steps:
            optimizer.zero_grad()
            local_model.load_state_dict(global_model.state_dict())

            dones = torch.zeros(params.unroll_steps, device=device)
            rewards = torch.zeros_like(dones, device=device)
            log_probs = torch.zeros_like(dones, device=device)
            entropies = torch.zeros_like(dones, device=device)
            values = torch.zeros_like(dones, device=device)

            if done:
                state, info = env.reset(seed=42 + rank)
                state = preprocessing(state)
                hx = torch.zeros(1, params.hidden_size, device=device)
                cx = torch.zeros(1, params.hidden_size, device=device)
            else:
                hx = hx.detach()
                cx = cx.detach()

            for step in range(params.unroll_steps):
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                policy, value, hx, cx = local_model(
                    state_tensor, action, reward, (hx, cx)
                )

                dist = Categorical(probs=policy)
                action = dist.sample()
                entropy = dist.entropy()
                log_prob = dist.log_prob(action)

                next_state, reward, done, _, info = env.step(action.cpu().item())
                next_state = preprocessing(next_state)
                pixel_change = pixel_diff(state, next_state)
                experience_replay.store(
                    state, reward, action.cpu().item(), done, pixel_change
                )

                values[step] = torch.squeeze(value)
                entropies[step] = entropy
                log_probs[step] = torch.squeeze(log_prob)
                dones[step] = torch.tensor(not done, device=device)
                rewards[step] = torch.tensor(reward, device=device)

                state = next_state
                action = F.one_hot(action, num_classes=env.action_space.n).to(device)
                reward = torch.tensor([[reward]], device=device).float()

                with global_steps.get_lock():
                    global_steps.value += 1

                if done:
                    with global_lines.get_lock():
                        global_lines.value += info["number_of_lines"]

                    if not rank:
                        writer.add_scalar(
                            f"Agent Block placed",
                            sum(info["statistics"].values()),
                            global_episodes.value,
                        )
                    break

            # Bootstrapping
            R = 0.0
            with torch.no_grad():
                _, R, _, _ = local_model(
                    torch.tensor(next_state, device=device).float().unsqueeze(0),
                    action,
                    reward,
                    (hx, cx),
                )

            # Hitung loss A3C
            actor_loss, critic_loss = local_model.a3c_loss(
                rewards=rewards[: step + 1],
                R=R,
                dones=dones[: step + 1],
                log_probs=log_probs[: step + 1],
                entropies=entropies[: step + 1],
                values=values[: step + 1],
            )
            a3c_loss = actor_loss + 0.5 * critic_loss
            episode_rewards = rewards.sum().cpu().detach()

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
            total_loss = a3c_loss + params.pc_weight * pc_loss + rp_loss + vr_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), params.grad_norm)
            ensure_share_grads(
                local_model=local_model, global_model=global_model
            )
            optimizer.step()

            if not rank:
                writer.add_scalar(f"Total Loss", total_loss, global_episodes.value)
                writer.add_scalar(f"Rewards", episode_rewards, global_episodes.value)
                writer.add_scalar(
                    f"Total lines cleared", global_lines.value, global_episodes.value
                )
                writer.add_scalar(
                    f"A3C Loss", a3c_loss.detach().cpu().numpy(), global_episodes.value
                )
                writer.add_scalar(
                    f"PC Loss", pc_loss.detach().cpu().numpy(), global_episodes.value
                )
                writer.add_scalar(
                    f"RP Loss", rp_loss.detach().cpu().numpy(), global_episodes.value
                )
                writer.add_scalar(
                    f"VR Loss", vr_loss.detach().cpu().numpy(), global_episodes.value
                )
                writer.add_scalar(
                    f"Entropy",
                    entropies.detach().mean().cpu().numpy(),
                    global_episodes.value,
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
                        "lines": global_lines.value,
                    },
                    f"{params.model_path}/UNREAL-heuristic_checkpoint.tar",
                )

        if not rank:
            torch.save(
                global_model.state_dict(), f"{params.model_path}/UNREAL-heuristic.pt"
            )
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
        if not rank:
            torch.save(
                {
                    "model_state_dict": global_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "steps": global_steps.value,
                    "episodes": global_episodes.value,
                    "lines": global_lines.value,
                },
                f"{params.model_path}/UNREAL-heuristic_checkpoint.tar",
            )
            writer.close()
        env.close()
        print(f"\tProses pelatihan agen {rank} dihentikan")
