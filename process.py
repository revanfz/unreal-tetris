import torch
import timeit
import pygame
import gymnasium as gym
import torch.nn.functional as F
import torchvision.transforms as T

import custom_env

from model import ActorCritic
from tensorboardX import SummaryWriter
from torch.distributions import Categorical

from optimizer import SharedAdam


def transformImage(image):
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Grayscale(num_output_channels=1),
            T.Resize((84, 84)),
        ]
    )

    return transform(image)


def local_train(
    index: int,
    opt: tuple,
    global_model: ActorCritic,
    optimizer: SharedAdam,
    global_steps,
    timestamp=False,
):
    torch.manual_seed(42 + index)
    if timestamp:
        start_time = timeit.default_timer()
    writer = SummaryWriter(opt.log_path)
    env = gym.make(
        "SmartTetris-v0", render_mode="human" if index == 0 else opt.render_mode
    )
    local_model = ActorCritic(4, env.action_space.n)
    if local_model.device != "cpu":
        local_model.cuda()
    local_model.train()

    obs, info = env.reset()
    obs = transformImage(obs["matrix_image"]).to(local_model.device)
    state = torch.zeros((opt.framestack, 84, 84), device=local_model.device)
    state[-1] = obs
    done = True
    curr_episode = 0

    while global_steps.value <= opt.max_steps:
        if timestamp:
            if curr_episode % opt.update_episode == 0 and curr_episode > 0:
                torch.save(
                    global_model.state_dict(), "{}/a3c_tetris.pt".format(opt.model_path)
                )
        local_model.load_state_dict(global_model.state_dict())

        log_probs = torch.zeros(opt.sync_steps, device=local_model.device)
        values = torch.zeros(opt.sync_steps, device=local_model.device)
        rewards = torch.zeros(opt.sync_steps, device=local_model.device)
        entropies = torch.zeros(opt.sync_steps, device=local_model.device)

        for step in range(opt.sync_steps):
            policy, value = local_model(state)
            probs = F.softmax(policy, dim=1)
            log_prob = F.log_softmax(policy, dim=1)
            entropy = -(probs * log_prob).sum(1, keepdim=True)

            m = Categorical(probs)
            action = m.sample().item()

            obs, reward, done, _, info = env.step(action)
            obs = transformImage(obs["matrix_image"]).to(local_model.device)
            state = torch.cat((state[1:], obs), dim=0).to(local_model.device)

            values[step] = value
            log_probs[step] = log_prob[0, action]
            rewards[step] = reward
            entropies[step] = entropy

            with global_steps.get_lock():
                global_steps.value += 1

            if done:
                values = values[: step + 1]
                log_probs = log_probs[: step + 1]
                rewards = rewards[: step + 1]
                entropies = entropies[: step + 1]
                break

        total_loss = local_model.calculate_loss(
            done, values, log_probs, entropies, rewards, opt.gamma, opt.beta
        )

        optimizer.zero_grad()
        total_loss.backward()

        for local_param, global_param in zip(
            local_model.parameters(), global_model.parameters()
        ):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        optimizer.step()

        writer.add_scalar("Train_Agent {}/Loss".format(index), total_loss, curr_episode)
        curr_episode += 1

        if done:
            writer.add_scalar(
                "Block Placed_Agent {}".format(index),
                info["block_placed"],
                curr_episode,
            )
            writer.add_scalar(
                "Lines Cleared_Agent {}".format(index),
                info["total_lines"],
                curr_episode,
            )
            obs, info = env.reset()
            obs = transformImage(obs["matrix_image"])
            state = torch.zeros((opt.framestack, 84, 84), device=local_model.device)
            state[-1] = obs

        print("Process {}. Finished episode {}".format(index, curr_episode))

        if global_steps.value >= opt.max_steps:
            print("Training process {} terminated".format(index))
            if timestamp:
                end_time = timeit.default_timer()
                print("The code runs for %.2f s " % (end_time - start_time))
            return


def local_test(index, opt, global_model):
    torch.manual_seed(42 + index)
    env = gym.make("SmartTetris-v0", render_mode="human")
    local_model = ActorCritic(4, env.action_space.n)
    local_model.eval()
    obs, info = env.reset()
    obs = transformImage(obs["matrix_image"])
    state = torch.zeros((opt.framestack, 84, 84))
    state[-1] = obs
    done = True
    while True:
        if done:
            local_model.load_state_dict(global_model.state_dict())

        policy, value = local_model(state)
        probs = F.softmax(policy, dim=1)
        action = torch.argmax(probs).item()
        obs, reward, done, _, info = env.step(action)
        obs = transformImage(obs["matrix_image"])
        state = torch.cat((state[1:], obs), dim=0)
        env.render()
        if done:
            obs, info = env.reset()
            obs = transformImage(obs["matrix_image"])
            state = torch.zeros((opt.framestack, 84, 84))
            state[-1] = obs
