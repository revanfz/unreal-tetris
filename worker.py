from multiprocessing.managers import SyncManager
import gym_tetris
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import UNREAL
from argparse import Namespace
from optimizer import SharedAdam
from replay_buffer import ReplayBuffer
from wrapper import ActionRepeatWrapper
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
from torch.distributions import Categorical
from multiprocessing.sharedctypes import Synchronized
from torch.utils.tensorboard import SummaryWriter
from utils import ensure_share_grads, preprocess_frame_stack


def worker(
    rank: int,
    global_model: UNREAL,
    optimizer: SharedAdam,
    shared_replay_buffer: ReplayBuffer,
    global_steps: Synchronized,
    global_episodes: Synchronized,
    shared_dict: SyncManager,
    params: Namespace,
    agent_episodes: int = 0,
):
    try:
        finished = False
        torch.manual_seed(42 + rank)
        if not rank:
            writer = SummaryWriter(params.log_path)

        env = gym_tetris.make(
            "TetrisA-v3",
            apply_api_compatibility=True,
            render_mode="human" if not rank else "rgb_array",
        )
        env = JoypadSpace(env, MOVEMENT)
        env = ActionRepeatWrapper(env)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_model = UNREAL(
            n_inputs=(3, 84, 84),
            n_actions=env.action_space.n,
            hidden_size=256,
            device=device,
        )
        local_model.train()
        local_replay_buffer = ReplayBuffer(params.unroll_steps)

        done = True

        while global_steps.value <= params.max_steps:
            optimizer.zero_grad()
            local_model.load_state_dict(global_model.state_dict())

            if done:
                state, info = env.reset()
                state = preprocess_frame_stack(state)
                local_replay_buffer.clear()
                prev_action = torch.zeros(1, env.action_space.n).to(device)
                prev_reward = torch.zeros(1, 1).to(device)
                hx = torch.zeros(1, params.hidden_size).to(device)
                cx = torch.zeros(1, params.hidden_size).to(device)
                episode_reward = 0
            else:
                hx = hx.data
                cx = cx.data

            for _ in range(params.unroll_steps):
                if not rank:
                    env.render()
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                policy, _, _, _, (hx, cx) = local_model(
                    state_tensor, prev_action, prev_reward, (hx, cx)
                )

                dist = Categorical(policy)
                action = dist.sample().detach()

                next_state, reward, done, _, info = env.step(action.item())
                next_state = preprocess_frame_stack(next_state)

                prev_action = F.one_hot(action, num_classes=env.action_space.n).to(device)
                prev_reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
                shared_replay_buffer.push(
                    state, action.item(), reward, next_state, done
                )
                local_replay_buffer.push(
                    state, action.item(), reward, next_state, done
                )

                episode_reward += reward
                state = next_state
                with global_steps.get_lock():
                    global_steps += 1

                if done:
                    break

            # Hitung loss A3C
            states, actions, rewards, _, dones = local_replay_buffer.sample(params.unroll_steps)
            policy_loss, value_loss = local_model.a3c_loss(states, rewards, dones, actions)
            a3c_loss = policy_loss + value_loss

            # Hitung Loss Pixel Control dan Feature Control
            states, actions, rewards, next_states, dones = shared_replay_buffer.sample(params.unroll_steps)
            aux_control_loss = local_model.control_loss(states, actions, rewards, next_states, dones)

            # Hitung Loss Reward Pedictions
            states, rewards = shared_replay_buffer.sample_rp(3)
            rp_loss = local_model.rp_loss(states, rewards)

            # Hitung value replay loss
            states, actions, rewards, _, dones = shared_replay_buffer.sample(params.unroll_steps)
            vr_loss = local_model.vr_loss(states, actions, rewards, dones)

            total_loss = a3c_loss + vr_loss + params.task_weight * aux_control_loss + rp_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), 40)
            ensure_share_grads(
                local_model=local_model, global_model=global_model, device=device
            )
            optimizer.step()

            if not rank:
                writer.add_scalar(
                    f"Losses",
                    total_loss,
                    global_episodes.value
                )

                writer.add_scalar(
                    f"Rewards",
                    episode_reward,
                    global_episodes.value
                )

                writer.add_scalar(
                    f"Lines cleared",
                    info["number_of_lines"],
                    global_episodes.value
                )

                writer.add_scalar(
                    f"Block placed",
                    sum(info["statistics"].values()),
                    global_episodes.value
                )

                if global_episodes % 100 == 0:
                    writer.flush()

            agent_episodes += 1
            with global_episodes.get_lock():
                global_episodes.value += 1
        
        if not rank:
            torch.save(
                global_model.state_dict(),
                f"{params.model_path}/a3c_tetris.pt"
            )
        finished = True
        print("Pelatihan agen {rank} selesai")
    
    except KeyboardInterrupt as e:
        print(f"Program dihentikan")

    except torch.multiprocessing.ProcessError as e:
        print("Unexpected error")

    finally:
        if not finished and not rank:
            torch.save({
                "model_state_dict": global_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "steps": global_steps.value.value,
                "episodes": global_episodes.value
            }, f"{params.model_path}/a3c_checkpoint.tar")
        env.close()
        writer.close()
        shared_dict[f"agent_{rank}"] = agent_episodes
        print(f"Proses pelatihan agen {rank} dihentikan")
