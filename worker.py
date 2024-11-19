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
    global_scores: Synchronized,
    params: Namespace,
    device: torch.device,
):
    try:
        finished = False
        torch.manual_seed(42 + rank)

        render_mode = "human" if not rank else "rgb_array"
        env = make_env(
            resize=84, render_mode=render_mode, level=19, id="TetrisA-v2", skip=2
        )

        # device = torch.device("cuda")
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
            writer = SummaryWriter(f"{params.log_path}/final")
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
        hx = torch.zeros(1, params.hidden_size).to(device)
        cx = torch.zeros(1, params.hidden_size).to(device)

        while not experience_replay._is_full():
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                policy, _, _, _ = local_model(
                    state_tensor, prev_action, prev_reward, None
                )

                dist = Categorical(probs=policy)
                action = dist.sample().cpu()

            next_state, reward, done, _, info = env.step(action.item())
            # if info["number_of_lines"] > last_lines:
            #     reward += (info["number_of_lines"] - last_lines) ** 2 * 10
            #     last_lines = info["number_of_lines"]
            # reward += (sum(info["statistics"].values()) - last_block) * 1
            # last_block = sum(info["statistics"].values())

            next_state = preprocessing(next_state)
            pixel_change = pixel_diff(state, next_state)
            experience_replay.store(state, reward, action.item(), done, pixel_change)
            
            state = next_state
            prev_action = F.one_hot(action, num_classes=env.action_space.n).to(device)
            prev_reward = torch.FloatTensor([[reward]]).to(device)

            hx = hx.detach()
            cx = cx.detach()

            if done:
                state, info = env.reset()
                state = preprocessing(state)
                hx = torch.zeros(1, params.hidden_size).to(device)
                cx = torch.zeros(1, params.hidden_size).to(device)

        done = True
        current_episodes = 0

        while global_steps.value <= params.max_steps:
            optimizer.zero_grad()
            local_model.load_state_dict(global_model.state_dict())
            episode_length = 0

            if done:
                state, info = env.reset()
                state = preprocessing(state)
                action = F.one_hot(torch.LongTensor([0]), num_classes=env.action_space.n).to(device)
                reward = torch.zeros(1, 1).to(device)
                hx = torch.zeros(1, params.hidden_size).to(device)
                cx = torch.zeros(1, params.hidden_size).to(device)
            else:
                hx = hx.detach()
                cx = cx.detach()

            for _ in range(params.unroll_steps):
                state_tensor =  torch.from_numpy(state).unsqueeze(0).to(device)
                policy, _, hx, cx = local_model(
                    state_tensor, action, reward, (hx, cx)
                )

                dist = Categorical(probs=policy)
                action = dist.sample()

                next_state, reward, done, _, info = env.step(action.item())
                # if info["number_of_lines"] > last_lines:
                #     reward += (info["number_of_lines"] - last_lines) ** 2 * 10
                #     last_lines = info["number_of_lines"]
                # reward += (sum(info["statistics"].values()) - last_block) * 1
                # last_block = sum(info["statistics"].values())

                next_state = preprocessing(next_state)
                pixel_change = pixel_diff(state, next_state)
                experience_replay.store(
                    state, reward, action.item(), done, pixel_change
                )
                state = next_state

                action = F.one_hot(action, num_classes=env.action_space.n).to(
                    device
                )
                reward = torch.FloatTensor([[reward]]).to(device)

                episode_length += 1

                with global_steps.get_lock():
                    global_steps.value += 1

                if done:
                    with global_lines.get_lock():
                        global_lines.value += info["number_of_lines"]

                    with global_scores.get_lock():
                        global_scores.value += info["score"]
                    
                    if not rank:
                        writer.add_scalar(
                            f"Agent Block placed",
                            sum(info["statistics"].values()),
                            global_episodes.value,
                        )

                        writer.add_scalar(
                            f"Agent Scores",
                            info["score"],
                            global_episodes.value,
                        )

                        writer.add_scalar(
                            f"Agent Lines",
                            info['number_of_lines'],
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

            # Hitung loss A3C
            states, rewards, actions, dones, pixel_change = experience_replay.sample(episode_length)
            a3c_loss, entropy = local_model.a3c_loss(
                states=states,
                dones=dones,
                actions=actions,
                rewards=rewards,
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

            # print(f"A3C Loss = {a3c_loss}\t PC Loss = {pc_loss}\t VR Loss = {vr_loss}\t RP Loss = {rp_loss}\n" )

            # Penjumlahan loss a3c, pixel control, value replay dan reward prediction
            total_loss = (
                a3c_loss + params.pc_weight * pc_loss + rp_loss + vr_loss 
            )

            total_loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), 40)
            ensure_share_grads(
                local_model=local_model, global_model=global_model, device=device
            )
            optimizer.step()

            if not rank:
                writer.add_scalar(f"Losses", total_loss, global_episodes.value)
                writer.add_scalar(f"Rewards", sum(rewards), global_episodes.value)
                writer.add_scalar(
                    f"Total lines cleared", global_lines.value, global_episodes.value
                )
                writer.add_scalar(
                    f"Total scores", global_scores.value, global_episodes.value
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
                            "lines": global_lines.value,
                            "scores": global_scores.value,
                        },
                        f"{params.model_path}/final_checkpoint.tar",
                    )

        if not rank:
            torch.save(global_model.state_dict(), f"{params.model_path}/final.pt")
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
                    "lines": global_lines.value,
                    "scores": global_scores.value,
                },
                f"{params.model_path}/final_checkpoint.tar",
            )
        if not rank:
            writer.close()
        env.close()
        print(f"Proses pelatihan agen {rank} dihentikan")
