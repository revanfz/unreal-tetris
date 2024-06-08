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


def transformImage(image):
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Grayscale(num_output_channels=1),
            T.Resize((84, 84)),
        ]
    )

    
    new_image = transform(image)
    mean = new_image.mean()
    std = new_image.std()

    normalize = T.Compose(
        [
            T.Normalize(mean, std)
        ]
    )

    new_image = normalize(new_image)
    return new_image


def local_train(index, opt, global_model, optimizer, global_eps, timestamp=False):
    torch.manual_seed(42 + index)
    if timestamp:
        start_time = timeit.default_timer()
    writer = SummaryWriter(opt.log_path)
    env = gym.make("SmartTetris-v0", render_mode=opt.render_mode)
    local_model = ActorCritic(4, env.action_space.n)
    if torch.cuda.is_available():
        local_model.cuda()
    local_model.train()
    obs, info = env.reset()
    obs = transformImage(obs["matrix_image"])
    state = torch.zeros((opt.framestack, 84, 84))
    state[0] = obs
    if torch.cuda.is_available():
        state = state.cuda()
    done = True
    curr_episode = 0
    while True:
        if timestamp:
            if curr_episode % opt.update_episode == 0 and curr_episode > 0:
                torch.save(
                    global_model.state_dict(), "{}/a3c_tetris.pt".format(opt.model_path)
                )
        local_model.load_state_dict(global_model.state_dict())

        log_probs = torch.zeros(
            opt.sync_steps, 1, device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        values = torch.zeros(
            opt.sync_steps, 1, device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        rewards = torch.zeros(
            opt.sync_steps, 1, device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        entropies = torch.zeros(
            opt.sync_steps, 1, device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        masks = torch.zeros(
            opt.sync_steps, 1, device="cuda:0" if torch.cuda.is_available() else "cpu"
        )

        for step in range(opt.sync_steps):
            policy, value = local_model(state)
            probs = F.softmax(policy, dim=1)
            log_prob = F.log_softmax(policy, dim=1)
            entropy = -(probs * log_prob).sum(1, keepdim=True)

            m = Categorical(probs)
            action = m.sample().item()

            obs, reward, done, _, info = env.step(action)
            obs = transformImage(obs["matrix_image"])
            if torch.cuda.is_available():
                obs = obs.cuda()
            state = torch.cat((state[1:], obs), dim=0)

            values[step] = value
            log_probs[step] = log_prob[0, action]
            rewards[step] = reward
            entropies[step] = entropy
            masks[step] = int(not done)

            if done:
                writer.add_scalar(
                    "Score_Agent {}".format(index), info["score"], curr_episode
                )
                writer.add_scalar(
                    "Lines Cleared_Agent {}".format(index),
                    info["lines_cleared"],
                    curr_episode,
                )
                obs, info = env.reset()
                obs = transformImage(obs["matrix_image"])
                state = torch.zeros((opt.framestack, 84, 84))
                state[0] = obs
                if torch.cuda.is_available():
                    state = state.cuda()
                break

        gae = 0.0
        advantages = torch.zeros(len(rewards))
        if torch.cuda.is_available():
            advantages = advantages.cuda()

        for t in reversed(range(len(rewards) - 1)):
            td_error = rewards[t] + opt.gamma * masks[t] * values[t + 1] - values[t]
            gae = td_error + opt.gamma * masks[t] * gae
            advantages[t] = gae

        critic_loss = advantages.pow(2).mean()
        actor_loss = (
            -(advantages.detach() * log_probs).mean() - opt.beta * entropy.mean()
        )
        total_loss = actor_loss + critic_loss * 0.5
        writer.add_scalar("Train_Agent {}/Loss".format(index), total_loss, curr_episode)
        writer.add_scalar("Train All Agents/Loss", total_loss, global_eps.value)
        optimizer.zero_grad()
        total_loss.backward()

        for local_param, global_param in zip(
            local_model.parameters(), global_model.parameters()
        ):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        optimizer.step()

        
        curr_episode += 1
        with global_eps.get_lock():
            global_eps.value += 1
        print("Process {}. Finished episode {}".format(index, curr_episode))

        if global_eps.value == opt.max_episode:
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
    state[0] = obs
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
            state[0] = obs
