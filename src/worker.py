import torch

from argparse import Namespace
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from multiprocessing.sharedctypes import Synchronized

from src.utils import make_env
from src.optimizer import SharedAdam
from src.model import ActorCriticNetwork


def worker(
    rank: int,
    global_model: ActorCriticNetwork,
    optimizer: SharedAdam,
    global_steps: Synchronized,
    global_episodes: Synchronized,
    params: Namespace,
    device: torch.device,
) -> None:
    try:
        torch.manual_seed(rank + 42)
        finished = False
        render_mode = "rgb_array" if rank else "human"
        env = make_env(render_mode=render_mode)

        local_model = ActorCriticNetwork(
            input_channels=env.observation_space.shape[0],
            num_actions=env.action_space.n,
            device=device,
            hidden_size=params.hidden_size,
        )
        local_model.train()

        if not rank:
            log_writer = SummaryWriter(
                f"{params.log_path}/tetris_a3c"
            )

            log_writer.add_graph(
                local_model,
                (
                    torch.zeros(1, 4, 84, 84),
                    (
                        torch.zeros(1, params.hidden_size),
                        torch.zeros(1, params.hidden_size),
                    ),
                ),
            )

        done = True

        while global_steps.value <= params.max_steps:
            if done:
                obs, info = env.reset()
                hx = torch.zeros(1, params.hidden_size).to(device)
                cx = torch.zeros(1, params.hidden_size).to(device)
            else:
                hx = hx.data
                cx = cx.data

            local_model.zero_grad()
            local_model.load_state_dict(global_model.state_dict())

            episode_rewards = 0
            values, log_probs, rewards, entropies = [], [], [], []

            for _ in range(int(params.rollout_steps)):
                state = torch.FloatTensor(obs.__array__().squeeze(3)).unsqueeze(0)
                policy, value, (hx, cx) = local_model(state, (hx, cx))

                dist = Categorical(policy)
                action = dist.sample().detach()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                obs, reward, done, _, info = env.step(action.item())
                episode_rewards += reward

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)

                with global_steps.get_lock():
                    global_steps.value += 1

                if global_steps.value % params.checkpoint_steps == 0:
                    torch.save(
                        {
                            "model_state_dict": global_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "steps": global_steps.value,
                            "episodes": global_episodes.value,
                        },
                        f"{params.model_path}/a3c_checkpoint.tar",
                    )

                if done:
                    break

            with global_episodes.get_lock():
                global_episodes.value += 1

            R = torch.zeros(1, 1).to(device)
            gae = torch.zeros(1, 1).to(device)

            if not done:
                bootstrap_state = torch.FloatTensor(
                    obs.__array__().squeeze(3)
                ).unsqueeze(0)
                with torch.no_grad():
                    _, value, _ = local_model(bootstrap_state, (hx, cx))
                R = value.data
            values.append(R)

            actor_loss = 0
            critic_loss = 0

            for t in reversed(range(len(rewards))):
                R = params.gamma * R + rewards[t]
                advantage = R - values[t]
                critic_loss = critic_loss + advantage.pow(2) * 0.5

                delta_t = (
                    rewards[t] + params.gamma * values[t + 1].data - values[t].data
                )
                gae = gae * params.gamma + delta_t
                actor_loss = (
                    actor_loss - log_probs[t] * gae - params.beta * entropies[t]
                )

            total_loss = actor_loss + 0.5 * critic_loss
            total_loss.backward()

            for local_param, global_param in zip(
                local_model.parameters(), global_model.parameters()
            ):
                if global_param.grad is not None:
                    break
                if device.type == "cuda":
                    global_param._grad = local_param.grad.cpu()
                else:
                    global_param._grad = local_param.grad
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
            optimizer.step()

            if not rank:
                log_writer.add_scalar(f"Losses", total_loss, global_episodes.value)

                log_writer.add_scalar(
                    f"Rewards", episode_rewards / params.rollout_steps, global_episodes.value
                )

                log_writer.add_scalar(
                    f"Lines cleared", info["number_of_lines"], global_episodes.value
                )

                log_writer.add_scalar(
                    f"Entropy", sum(entropies) / len(entropies), global_episodes.value
                )

                log_writer.add_scalar(
                    f"Block placed",
                    sum(info["statistics"].values()),
                    global_episodes.value,
                )
        if not rank:
            torch.save(global_model.state_dict(), f"{params.model_path}/a3c_tetris.pt")

    except KeyboardInterrupt as e:
        print(f"Program dihentikan")
        raise KeyboardInterrupt("Program dihentikan")

    except torch.multiprocessing.ProcessError as e:
        print("Unexpected error")
        raise Exception(f"Multiprocessing error\t{e}.")

    except Exception as e:
        print(f"Error ;X\n{e}")
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
            log_writer.close()
        env.close()
        print(f"Proses pelatihan agen {rank} dihentikan")
