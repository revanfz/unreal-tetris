import torch
import timeit
import pygame
import gymnasium as gym
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as utils

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

    image = transform(image)
    utils.save_image(image, "test.jpg")

    return image


def local_train(index, opt, global_model, optimizer, timestamp=False):
    torch.manual_seed(42 + index)
    if timestamp:
        start_time = timeit.default_timer()
    writer = SummaryWriter(opt.log_path)
    env = gym.make("SmartTetris-v0", render_mode=opt.render_mode)
    local_model = ActorCritic(1, env.action_space.n)
    if torch.cuda.is_available():
        local_model.cuda()
    local_model.train()
    state, info = env.reset()
    state = transformImage(state["matrix_image"])
    if torch.cuda.is_available():
        state = state.cuda()
    done = True
    curr_episode = 0
    while True:
        if timestamp:
            if curr_episode % opt.update_episode == 0 and curr_episode > 0:
                torch.save(
                    global_model.state_dict(), "{}/a3c_tetris".format(opt.model_path)
                )
        local_model.load_state_dict(global_model.state_dict())

        if done:
            hx = torch.zeros((1, 256), dtype=torch.float)
            cx = torch.zeros((1, 256), dtype=torch.float)
        else:
            hx = hx.detach()
            cx = cx.detach()
        if torch.cuda.is_available():
            hx = hx.cuda()
            cx = cx.cuda()

        log_probs = []
        values = []
        rewards = []
        entropies = []

        for _ in range(opt.sync_steps):
            policy, value, hx, cx = local_model(state, hx, cx)
            probs = F.softmax(policy, dim=1)
            log_prob = F.log_softmax(policy, dim=1)
            entropy = -(probs * log_prob).sum(1, keepdim=True)

            m = Categorical(probs)
            action = m.sample().item()

            state, reward, done, _, info = env.step(action)
            state = transformImage(state["matrix_image"])
            if torch.cuda.is_available():
                state = state.cuda()

            values.append(value)
            log_probs.append(log_prob[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                curr_episode += 1
                writer.add_scalar("Score_Agent {}".format(index), info["score"], curr_episode)
                writer.add_scalar("Lines Cleared_Agent {}".format(index), info["lines_cleared"], curr_episode)
                state, info = env.reset()
                state = transformImage(state["matrix_image"])
                if torch.cuda.is_available():
                    state = state.cuda()

            if done:
                print("Process {}. Episode {}".format(index, curr_episode))
                print("Rewards {}, Episode {}".format(sum(rewards), curr_episode))
                break

        R = torch.zeros((1, 1), dtype=torch.float)
        if torch.cuda.is_available():
            R = R.cuda()
        if not done:
            _, R, _, _ = local_model(state, hx, cx)

        advantage = torch.zeros((1, 1), dtype=torch.float)
        if torch.cuda.is_available():
            advantage = advantage.cuda()

        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R

        for value, log_policy, reward, entropy in list(
            zip(values, log_probs, rewards, entropies)
        )[::-1]:
            advantage = advantage * opt.gamma
            td_error = next_value.detach() - value.detach()
            advantage = advantage + reward + opt.gamma * td_error
            next_value = value
            actor_loss = actor_loss + log_policy * advantage
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
        writer.add_scalar("Train_Agent {}/Loss".format(index), total_loss, curr_episode)
        optimizer.zero_grad()
        total_loss.backward()

        for local_param, global_param in zip(
            local_model.parameters(), global_model.parameters()
        ):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        optimizer.step()

        if curr_episode == opt.max_episode:
            print("Training process {} terminated".format(index))
            if timestamp:
                end_time = timeit.default_timer()
                print("The code runs for %.2f s " % (end_time - start_time))
            return


def local_test(index, opt, global_model):
    torch.manual_seed(42 + index)
    env = gym.make("SmartTetris-v0", render_mode="human")
    local_model = ActorCritic(1, env.action_space.n)
    local_model.eval()
    state, info = env.reset()
    state = transformImage(state["matrix_image"])
    done = True
    while True:
        if done:
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                hx = torch.zeros((1, 256), dtype=torch.float)
                cx = torch.zeros((1, 256), dtype=torch.float)
            else:
                hx = hx.detach()
                cx = cx.detach()

        policy, value, hx, cx = local_model(state, hx, cx)
        probs = F.softmax(policy, dim=1)
        action = torch.argmax(probs).item()
        state, reward, done, _, info = env.step(action)
        env.render()
        if done:
            state, info = env.reset()
        state = transformImage(state["matrix_image"])
