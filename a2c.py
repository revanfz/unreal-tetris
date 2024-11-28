import cv2
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from model import UNREAL
from optimizer import SharedAdam
from torchvision.transforms import v2
from replay_buffer import ReplayBuffer
from torch.distributions import Categorical
from utils import make_env, batch_pixel_diff
from torch.utils.tensorboard import SummaryWriter


def preprocessing_state(x: torch.Tensor, device: torch.device):
    preprocess = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    return preprocess(x).to(device)


def batch_samples(samples: np.ndarray, device: torch.device):
    states, rewards, actions, dones, pixel_changes = [], [], [], [], []
    for env_sample in samples:
        states.append(env_sample[0])  # shape: [seq_len, channels, height, width]
        rewards.append(env_sample[1])  # shape: [seq_len]
        actions.append(env_sample[2])  # shape: [seq_len]
        dones.append(env_sample[3])  # shape: [seq_len]
        pixel_changes.append(env_sample[4])  # shape: [seq_len, 20, 20]

    states_batch = torch.tensor(
        np.array(states), device=device
    ).float()  # [n_envs, seq_len, channels, height, width]
    rewards_batch = torch.tensor(np.array(rewards), device=device).unsqueeze(
        -1
    )  # [n_envs, seq_len, 1]
    actions_batch = torch.tensor(
        np.array(actions), device=device
    ).long()  # [n_envs, seq_len]
    dones_batch = torch.tensor(np.array(dones), device=device).unsqueeze(
        -1
    )  # [n_envs, seq_len]
    pixel_changes_batch = torch.tensor(
        np.array(pixel_changes), device=device
    ).float()  # [n_envs, seq_len, 20, 20]

    return states_batch, rewards_batch, actions_batch, dones_batch, pixel_changes_batch


def train_a2c(
    n_envs: int = 8,
    unroll_steps: int = 20,
    device: str = "cpu",
    n_updates: int = 100_000,
    beta: float = 0.001,
    gamma: float = 0.99,
    hidden_size: int = 256,
    learning_rate: float = 1e-3,
    log_path: str = "./tensorboard",
    log: bool = False,
    plot: bool = False,
    tuning: bool = False
):
    try:
        torch.backends.cudnn.benchmark = True
        torch.autograd.set_detect_anomaly(True)
        device = torch.device(device)
        torch.manual_seed(42)
        np.random.seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)

        finished = False

        obs_space = (3, 84, 84)

        if log:
            writer = SummaryWriter(f"{log_path}/A2C-unreal/64")

        envs = gym.vector.AsyncVectorEnv(
            [
                lambda: make_env(
                    id="TetrisA-v3",
                    render_mode="human" if i == 0 else "rgb_array",
                    level=19,
                    skip=2,
                    resize=84,
                )
                for i in range(n_envs)
            ]
        )

        model = UNREAL(
            n_inputs=envs.single_observation_space.shape,
            n_actions=envs.single_action_space.n,
            hidden_size=hidden_size,
            device=device,
            beta=beta,
            gamma=gamma,
        )
        optimizer = SharedAdam(model.parameters(), lr=learning_rate)

        scaler = torch.GradScaler()

        envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(
            envs, buffer_length=n_envs * n_updates
        )
        state, info = envs_wrapper.reset(seed=42)
        state = preprocessing_state(torch.from_numpy(state.transpose(0, 3, 1, 2)), device)

        action = F.one_hot(
            torch.tensor([0 for _ in range(n_envs)], device=device).long(), model.n_actions
        )
        reward = torch.zeros(n_envs).to(device)

        hx = torch.zeros(n_envs, hidden_size).to(device)
        cx = torch.zeros(n_envs, hidden_size).to(device)

        if log:
            with torch.no_grad():
                writer.add_graph(
                    model,
                    (
                        state,
                        action,
                        reward.unsqueeze(1),
                        (hx, cx),
                    ),
                )

        replay_buffer = [ReplayBuffer(2000) for _ in range(n_envs)]

        # print("Filling experience...")
        # while not (all(buffer._is_full() for buffer in replay_buffer)):
        for step in tqdm(range(300), desc="Filling experience"):
            with torch.no_grad():
                policy, value, next_hx, next_cx = model(
                    state, action, reward.unsqueeze(1), (hx, cx)
                )

                dist = Categorical(probs=policy)
                action = dist.sample()

            next_state, reward, done, _, info = envs_wrapper.step(action.cpu().numpy())
            next_state = preprocessing_state(
                torch.from_numpy(next_state.transpose(0, 3, 1, 2)), device
            )
            pixel_change = batch_pixel_diff(state, next_state)

            for i in range(n_envs):
                replay_buffer[i].store(
                    state=state[i].detach().cpu().numpy(),
                    reward=reward[i],
                    action=action[i].detach().cpu().numpy(),
                    done=done[i],
                    pixel_change=pixel_change[i].detach().cpu().numpy(),
                )

                next_hx = next_hx.clone()
                next_cx = next_cx.clone()
                if done[i]:
                    next_hx[i] = torch.zeros_like(next_hx[i], device=device)
                    next_cx[i] = torch.zeros_like(next_cx[i], device=device)

            state = next_state
            action = F.one_hot(action.cpu().long(), num_classes=model.n_actions).to(device)
            reward = torch.tensor(reward, device=device)
            mask = torch.tensor([not d for d in done], device=device)
            hx = next_hx
            cx = next_cx

        print("Buffer filled.")

        # if plot:
        critic_losses = []
        actor_losses = []
        pc_losses = []
        vr_losses = []
        rp_losses = []
        entropies = []

        state, info = envs_wrapper.reset(seed=42)
        state = preprocessing_state(torch.from_numpy(state.transpose(0, 3, 1, 2)), device)
        prev_action = F.one_hot(
            torch.tensor([0 for _ in range(n_envs)], device=device).long(), model.n_actions
        )
        prev_reward = torch.zeros(n_envs, device=device)
        hx = torch.zeros(n_envs, hidden_size, device=device)
        cx = torch.zeros(n_envs, hidden_size, device=device)

        episode_rewards = 0
        log_count = 0

        for sample_phase in tqdm(range(n_updates), desc="Training model..."):
            with torch.autocast(device_type=device.type):
                optimizer.zero_grad()
                states = torch.zeros(unroll_steps, n_envs, *obs_space, device=device)
                actions = torch.zeros(unroll_steps, n_envs, model.n_actions, device=device)
                rewards = torch.zeros(unroll_steps, n_envs, device=device)
                pixel_changes = torch.zeros(unroll_steps, n_envs, 20, 20, device=device)
                mask = torch.zeros(unroll_steps, n_envs, device=device)
                values = torch.zeros(unroll_steps, n_envs, device=device)
                log_probs = torch.zeros(unroll_steps, n_envs, device=device)
                entropy = torch.zeros(unroll_steps, n_envs, device=device)

                hx = hx.detach()
                cx = cx.detach()

                for step in range(unroll_steps):
                    # frame_bgr = cv2.cvtColor(envs_wrapper.render()[0], cv2.COLOR_RGB2BGR)
                    # cv2.imshow("Tetris", frame_bgr)
                    policy, value, next_hx, next_cx = model(
                        state, prev_action, prev_reward.unsqueeze(1), (hx, cx)
                    )

                    dist = Categorical(probs=policy)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                    next_state, reward, done, _, info = envs_wrapper.step(
                        action.cpu().numpy()
                    )
                    next_state = preprocessing_state(
                        torch.from_numpy(next_state.transpose(0, 3, 1, 2)), device
                    )
                    pixel_change = batch_pixel_diff(state, next_state)

                    next_hx = next_hx.clone()
                    next_cx = next_cx.clone()
                    for i in range(n_envs):
                        replay_buffer[i].store(
                            state=state[i].detach().cpu().numpy(),
                            reward=reward[i],
                            action=action[i].detach().cpu().numpy(),
                            done=done[i],
                            pixel_change=pixel_change[i].detach().cpu().numpy(),
                        )

                        if done[i]:
                            next_hx[i] = torch.zeros_like(next_hx[i], device=device)
                            next_cx[i] = torch.zeros_like(next_cx[i], device=device)

                    states[step] = next_state
                    actions[step] = F.one_hot(
                        action.cpu().long(), num_classes=model.n_actions
                    ).to(device)
                    pixel_changes[step] = pixel_change
                    rewards[step] = torch.tensor(reward, device=device)
                    mask[step] = torch.tensor([not d for d in done], device=device)
                    values[step] = torch.squeeze(value)
                    log_probs[step] = torch.squeeze(log_prob)
                    entropy[step] = dist.entropy()

                    state = next_state
                    hx = next_hx
                    cx = next_cx

                    prev_action = actions[step]
                    prev_reward = rewards[step]

                    # if cv2.waitKey(1) & 0xFF == ord("q"):
                    #     break

                with torch.no_grad():
                    _, R, _, _ = model(
                        state, prev_action, prev_reward.unsqueeze(1), (hx, cx)
                    )
                    R = R.squeeze()

                actor_loss, critic_loss = model.a2c_loss(
                    R, rewards, values, mask, log_probs, entropy
                )

                episode_rewards += rewards.mean().detach().cpu().numpy()

                # Pixel Control Loss
                samples = []
                for i in range(n_envs):
                    states, rewards, actions, dones, pixel_changes = replay_buffer[
                        i
                    ].sample_sequence(unroll_steps + 1)
                    samples.append([states, rewards, actions, dones, pixel_changes])

                (
                    states_batch,
                    rewards_batch,
                    actions_batch,
                    dones_batch,
                    pixel_changes_batch,
                ) = batch_samples(samples, device)

                pc_loss = model.batch_pc_loss(
                    states_batch,
                    rewards_batch,
                    actions_batch,
                    dones_batch,
                    pixel_changes_batch,
                )

                # Reward prediction loss
                samples = []
                for i in range(n_envs):
                    states, rewards, actions, dones, pixel_changes = replay_buffer[
                        i
                    ].sample_rp()
                    samples.append([states, rewards, actions, dones, pixel_changes])
                (
                    states_batch,
                    rewards_batch,
                    _,
                    _,
                    _,
                ) = batch_samples(samples, device)
                rp_loss = model.batch_rp_loss(states_batch, rewards_batch)

                # Value Replay Loss
                samples = []
                for i in range(n_envs):
                    states, rewards, actions, dones, pixel_changes = replay_buffer[
                        i
                    ].sample_sequence(unroll_steps + 1)
                    samples.append([states, rewards, actions, dones, pixel_changes])

                (states_batch, rewards_batch, actions_batch, dones_batch, _) = (
                    batch_samples(samples, device)
                )

                vr_loss = model.batch_vr_loss(
                    states_batch, rewards_batch, actions_batch, dones_batch
                )

                total_loss = actor_loss + 0.5 * critic_loss + pc_loss + rp_loss + vr_loss

            scaler.scale(total_loss).backward()
            # total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 40)
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            if log:
                writer.add_scalar(
                    f"Total loss", total_loss.detach().cpu().numpy(), sample_phase
                )
                writer.add_scalar(
                    f"Entropy", entropy.detach().mean().cpu().numpy(), sample_phase
                )
                writer.add_scalar(
                    f"Critic loss", critic_loss.detach().cpu().numpy(), sample_phase
                )
                writer.add_scalar(
                    f"Actor loss", actor_loss.detach().cpu().numpy(), sample_phase
                )
                writer.add_scalar(f"PC loss", pc_loss.detach().cpu().numpy(), sample_phase)
                writer.add_scalar(f"RP loss", rp_loss.detach().cpu().numpy(), sample_phase)
                writer.add_scalar(f"VR loss", vr_loss.detach().cpu().numpy(), sample_phase)

                if sample_phase % 50 == 0:
                    writer.add_scalar(f"50 Episode rewards", episode_rewards, log_count)
                    episode_rewards = 0
                    log_count += 1

            if plot:
                critic_losses.append(critic_loss.detach().cpu().numpy())
                actor_losses.append(actor_loss.detach().cpu().numpy())
                pc_losses.append(pc_loss.detach().cpu().numpy())
                rp_losses.append(rp_loss.detach().cpu().numpy())
                vr_losses.append(vr_loss.detach().cpu().numpy())
                entropies.append(entropy.detach().mean().cpu().numpy())

        cv2.destroyAllWindows()
        finished = True
        return (
            critic_losses,
            actor_losses,
            pc_losses,
            rp_losses,
            vr_losses,
            entropies,
            envs_wrapper.return_queue,
        )

    except KeyboardInterrupt as e:
        print("Error occured ", end="")
        print(f"{e}.")

    finally:
        if not tuning:
            if finished:
                torch.save(
                    model.state_dict(), f"./trained_models/a2c-unreal.pt"
                )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "log_count": log_count,
                    "epochs": sample_phase,
                },
                f"trained_models/a2c-unreal_checkpoint.tar",
            )
        else:
            return np.sum(envs_wrapper.return_queue.flatten()) / (sample_phase * unroll_steps)


def plot_training(
    entropies,
    pc_losses,
    rp_losses,
    critic_losses,
    vr_losses,
    unroll_steps,
    return_queue,
    actor_losses,
    n_envs,
):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 6))
    fig.suptitle(
        f"Training plots for {UNREAL.__class__.__name__} in the TetrisA-v3 environment \n \
                (n_envs={n_envs}, unroll_steps={unroll_steps})"
    )

    # episode return
    axs[0][0].set_title("Episode Returns")
    episode_returns_moving_average = (
        np.convolve(
            np.array(return_queue).flatten(),
            np.ones(unroll_steps),
            mode="valid",
        )
        / unroll_steps
    )
    axs[0][0].plot(
        np.arange(len(episode_returns_moving_average)) / n_envs,
        episode_returns_moving_average,
    )
    axs[0][0].set_xlabel("Number of episodes")

    # entropy
    axs[1][0].set_title("Entropy")
    entropy_moving_average = (
        np.convolve(np.array(entropies), np.ones(unroll_steps), mode="valid")
        / unroll_steps
    )
    axs[1][0].plot(entropy_moving_average)
    axs[1][0].set_xlabel("Number of updates")

    # value replay loss
    axs[2][0].set_title("VR Loss")
    vr_losses_moving_average = (
        np.convolve(np.array(vr_losses).flatten(), np.ones(unroll_steps), mode="valid")
        / unroll_steps
    )
    axs[2][0].plot(vr_losses_moving_average)
    axs[2][0].set_xlabel("Number of updates")

    # critic loss
    axs[0][1].set_title("Critic Loss")
    critic_losses_moving_average = (
        np.convolve(
            np.array(critic_losses).flatten(), np.ones(unroll_steps), mode="valid"
        )
        / unroll_steps
    )
    axs[0][1].plot(critic_losses_moving_average)
    axs[0][1].set_xlabel("Number of updates")

    # actor loss
    axs[1][1].set_title("Actor Loss")
    actor_losses_moving_average = (
        np.convolve(
            np.array(actor_losses).flatten(), np.ones(unroll_steps), mode="valid"
        )
        / unroll_steps
    )
    axs[1][1].plot(actor_losses_moving_average)
    axs[1][1].set_xlabel("Number of updates")

    # pixel control loss
    axs[2][1].set_title("PC Loss")
    pc_losses_moving_average = (
        np.convolve(np.array(pc_losses).flatten(), np.ones(unroll_steps), mode="valid")
        / unroll_steps
    )
    axs[2][1].plot(pc_losses_moving_average)
    axs[2][1].set_xlabel("Number of updates")

    # reward prediction loss
    axs[0][2].set_title("RP Loss")
    rp_losses_moving_average = (
        np.convolve(np.array(rp_losses).flatten(), np.ones(unroll_steps), mode="valid")
        / unroll_steps
    )
    axs[0][2].plot(rp_losses_moving_average)
    axs[0][2].set_xlabel("Number of updates")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n_envs = 4
    n_updates = 5000
    unroll_steps = 20
    beta = 0.01
    learning_rate = 1e-4
    device = "cuda"
    critic_loss, actor_loss, pc_loss, rp_loss, vr_loss, entropies, return_queue = (
        train_a2c(
            n_envs=n_envs,
            device=device,
            n_updates=n_updates,
            unroll_steps=unroll_steps,
            log=True,
            # plot=True,
            beta=beta,
            learning_rate=learning_rate,
        )
    )

    if len(entropies) > 0:
        plot_training(
            critic_losses=critic_loss,
            actor_losses=actor_loss,
            pc_losses=pc_loss,
            rp_losses=rp_loss,
            vr_losses=vr_loss,
            return_queue=return_queue,
            unroll_steps=unroll_steps,
            n_envs=n_envs,
            entropies=entropies,
        )
