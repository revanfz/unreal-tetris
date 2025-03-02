import wandb
import torch
import torch.nn as nn
import wandb.integration
import wandb.integration.gym
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
    level: int,
    global_model: UNREAL,
    optimizer: SharedAdam,
    global_steps: Synchronized,
    global_episodes: Synchronized,
    global_lines: Synchronized,
    params: Namespace,
    device: torch.device,
):
    try:
        render_mode = "rgb_array"
        env = make_env(
            id="TetrisA-v3",
            resize=84,
            render_mode=render_mode,
            level=level,
            skip=2,
            # record=True if not rank else False,
            log_every=500,
            episode=global_episodes.value,
        )

        local_model = UNREAL(
            n_inputs=(84, 84, 3),
            n_actions=env.action_space.n,
            hidden_size=params.hidden_size,
            device=device,
            beta=params.beta,
            gamma=params.gamma,
            pc=global_model.use_pc,
            vr=global_model.use_vr,
            rp=global_model.use_rp
        )
        local_model.load_state_dict(global_model.state_dict())
        local_model.train()
        experience_replay = ReplayBuffer(2000)

        prev_action = F.one_hot(torch.tensor([0]).long(), env.action_space.n).to(device)
        prev_reward = torch.zeros(1, 1, device=device)

        if not rank:
            wandb.tensorboard.patch(root_logdir=f"{params.log_path}/UNREAL-cont-fine-tuning", pytorch=True)
            wandb.init(
                project="UNREAL-cont-fine-tuning",
                config={
                    "learning_rate": params.lr,
                    "optimizer": params.optimizer,
                    "unroll_steps": params.unroll_steps,
                    "pc_weight": params.pc_weight,
                    "grad_norm": params.grad_norm,
                    "hidden_size": params.hidden_size,
                    "num_agents": params.num_agents,
                    "gamma": global_model.gamma,
                    "beta": global_model.beta,
                    "n_actions": global_model.n_actions,
                    "input_dim": global_model.n_inputs,
                },
                id=f"UNREAL",
                resume="allow",
                name=f"UNREAL",
                sync_tensorboard=True
            )
            wandb.watch(local_model, log="all", log_freq=params.save_interval)
            ep_writer = SummaryWriter(f"{params.log_path}/UNREAL-cont-fine-tuning/episode")
            game_writer = SummaryWriter(f"{params.log_path}/UNREAL-cont-fine-tuning/game")
            with torch.no_grad():
                ep_writer.add_graph(
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
                state, info = env.reset()
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
        # env.recording = True
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

            episode_rewards = 0

            if done:
                state, info = env.reset()
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
                episode_rewards += reward
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
                    if info["number_of_lines"]:
                        with global_lines.get_lock():
                            global_lines.value += info["number_of_lines"]

                    if not rank:
                        game_writer.add_scalar(
                            f"Block placed",
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
            total_loss = a3c_loss

            # Hitung Loss Pixel Control
            # 1.  Sampling replay buffer secara random
            if local_model.use_pc:
                states, rewards, actions, dones, pixel_changes = (
                    experience_replay.sample_sequence(params.unroll_steps + 1)
                )
                # 2. Hitung loss Pixel Control
                pc_loss = local_model.control_loss(
                    states, rewards, actions, dones, pixel_changes
                )
                total_loss += pc_loss * params.pc_weight

            if local_model.use_rp:
                # Hitung Loss Reward Prediction
                # 1. Sampel frame dengan peluang rewarding state = 0.5
                states, rewards, actions, dones, pixel_changes = (
                    experience_replay.sample_rp()
                )
                # 2. Hitung loss reward prediction
                rp_loss = local_model.rp_loss(states, rewards)
                total_loss += rp_loss

            if local_model.use_vr:
                # Hitung loss Value Replay
                states, rewards, actions, dones, pixel_changes = (
                    experience_replay.sample_sequence(params.unroll_steps + 1)
                )
                vr_loss = local_model.vr_loss(states, actions, rewards, dones)
                total_loss += vr_loss

            # Penjumlahan loss a3c, pixel control, value replay dan reward prediction
            # total_loss = a3c_loss + params.pc_weight * pc_loss + rp_loss + vr_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), params.grad_norm)
            ensure_share_grads(local_model=local_model, global_model=global_model)
            optimizer.step()

            if not rank:
                with global_lines.get_lock():
                    ep_writer.add_scalar(f"Total Loss", total_loss, global_episodes.value)
                    ep_writer.add_scalar(f"Rewards", episode_rewards, global_episodes.value)
                    ep_writer.add_scalar(
                        f"Total lines cleared", global_lines.value, global_episodes.value
                    )
                    ep_writer.add_scalar(
                        f"A3C Loss", a3c_loss.detach().cpu().numpy(), global_episodes.value
                    )
                    ep_writer.add_scalar(
                        f"PC Loss", pc_loss.detach().cpu().numpy(), global_episodes.value
                    )
                    ep_writer.add_scalar(
                        f"RP Loss", rp_loss.detach().cpu().numpy(), global_episodes.value
                    )
                    ep_writer.add_scalar(
                        f"VR Loss", vr_loss.detach().cpu().numpy(), global_episodes.value
                    )
                    ep_writer.add_scalar(
                        f"Entropy",
                        entropies.detach().mean().cpu().numpy(),
                        global_episodes.value,
                    )

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
                    f"{params.model_path}/UNREAL-cont-fine-tuning_checkpoint.tar",
                )

        if not rank:
            torch.save(
                global_model.state_dict(), f"{params.model_path}/UNREAL-cont-fine-tuning.pt"
            )
            torch.onnx.export(
                global_model,
                (state_tensor, prev_action, prev_reward, (hx, cx)),
                f"{params.model_path}/UNREAL-cont-fine-tuning.onnx",
                input_names=["input"],
            )
            wandb.save(
                "UNREAL-cont-fine-tuning.onnx",
            )
        print(f"Pelatihan agen {rank} selesai")

    except KeyboardInterrupt as e:
        print(f"Program dihentikan")
        raise KeyboardInterrupt("Program dihentikan")

    except torch.multiprocessing.ProcessError as e:
        print("Unexpected error")
        raise Exception(f"Multiprocessing error\t{e}.")

    except Exception as e:
        print("Error: ", e)
        raise Exception(f"Error: {e}")

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
                f"{params.model_path}/UNREAL-cont-fine-tuning_checkpoint.tar",
            )
            ep_writer.close()
            game_writer.close()
            wandb.finish()
        print(f"\tProses pelatihan agen {rank} dihentikan")
