import torch
import timeit
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

    return transform(image)


def calculate_loss(local_model, opt, states, batch_actions, batch_hx, batch_cx, done, rewards):
    policy, values, _, _ = local_model(states, batch_hx, batch_cx) # batchnya 5
    R = values[-1] * (1 - int(done))

    returns = []
    for reward in rewards[::-1]:
        R = reward * opt.gamma
        returns.append(R)
    returns.reverse()

    discounted_rewards = torch.tensor(returns, dtype=torch.float, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    advantage = discounted_rewards - values
    if torch.cuda.is_available():
        advantage.cuda()
    
    probs = torch.softmax(policy, dim=1)
    distribution = Categorical(probs)
    batch_actions = torch.tensor(batch_actions, dtype=torch.int, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    log_probs = distribution.log_prob(batch_actions)
    entropy = distribution.entropy()

    entropy_loss = opt.beta * entropy
    critic_loss = F.mse_loss(discounted_rewards, values.squeeze(1))
    actor_loss = -log_probs * advantage.detach() - entropy_loss

    total_loss = (actor_loss + critic_loss).mean()
    return total_loss


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

        rewards = []
        batch_actions = []
        batch_states = []
        batch_hx = []
        batch_cx = []

        for _ in range(opt.sync_steps):
            policy, value, hx, cx = local_model(state, hx, cx) # btachnya 1
            probs = F.softmax(policy, dim=1)

            m = Categorical(probs)
            action = m.sample().item()
            batch_actions.append(action)
            batch_cx.append(cx)
            batch_hx.append(hx)

            state, reward, done, _, info = env.step(action)
            state = transformImage(state["matrix_image"])
            if torch.cuda.is_available():
                state = state.cuda()

            rewards.append(reward)
            batch_states.append(state)
            if done:
                curr_episode += 1
                writer.add_scalar(
                    "Score_Agent {}".format(index), info["score"], curr_episode
                )
                writer.add_scalar(
                    "Block Placed_Agent {}".format(index), info["block_placed"], curr_episode
                )
                writer.add_scalar(
                    "Lines Cleared_Agent {}".format(index),
                    info["lines_cleared"],
                    curr_episode,
                )
                state, info = env.reset()
                state = transformImage(state["matrix_image"])
                if torch.cuda.is_available():
                    state = state.cuda()

            if done:
                print("Process {}: Episode {} done".format(index, curr_episode))
                break

        states = torch.stack(batch_states, dim=0)
        batch_cx = torch.stack(batch_cx, dim=1).squeeze(0)
        batch_hx = torch.stack(batch_hx, dim=1).squeeze(0)

        if torch.cuda.is_available():
            states.cuda()
            batch_cx.cuda()
            batch_hx.cuda()

        total_loss = calculate_loss(local_model, opt, states, batch_actions, batch_hx, batch_cx, done, rewards)

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
