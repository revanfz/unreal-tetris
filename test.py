import torch
import argparse
import gymnasium as gym
import torch.nn.functional as F

import custom_env

from model import ActorCritic
from process import transformImage


def get_args():
    parser = argparse.ArgumentParser(
    """
        Implementation model A3C: 
        IMPLEMENTASI ALGORITMA ASYNCHRONOUS ADVANTAGE ACTOR-CRITIC (A3C)
        UNTUK MENGHASILKAN AGEN CERDAS (STUDI KASUS: PERMAINAN TETRIS)
    """)
    parser.add_argument("--framestack", type=int, default=4)
    parser.add_argument("--model-path", type=str, default="trained_models")
    parser.add_argument("--output-path", type=str, default="output")
    args = parser.parse_args()
    return args

def test(opt):
    torch.manual_seed(42)
    env = gym.make("SmartTetris-v0", render_mode="human")
    model = ActorCritic(4, env.action_space.n)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(f"{opt.model_path}/a3c_tetris.pt"))
        model.cuda()
    else:
        model.load_state_dict(torch.load(f"{opt.mode_path}/a3c_tetris.pt", map_location=lambda storage, loc: storage))

    model.eval()
    obs, info = env.reset()
    obs = transformImage(obs["matrix_image"])
    state = torch.zeros((opt.framestack, 84, 84))
    if torch.cuda.is_available():
        state = state.cuda()
    state[0] = obs
    done = True
    while True:
        policy, value = model(state)
        probs = F.softmax(policy, dim=1)
        print(probs)
        action = torch.argmax(probs).item()
        action = int(action)
        obs, reward, done, _, info = env.step(action)
        obs = transformImage(obs["matrix_image"])
        if torch.cuda.is_available():
            obs = obs.cuda()
        state = torch.cat((state[1:], obs), dim=0)

        if done:
            obs, info = env.reset()
            obs = transformImage(obs["matrix_image"])
            state = torch.zeros((opt.framestack, 84, 84))
            state[0] = obs
        
        if torch.cuda.is_available():
            state = state.cuda()

if __name__ == "__main__":
    opt = get_args()
    test(opt)