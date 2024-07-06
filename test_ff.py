from time import sleep
import torch
import argparse
import gymnasium as gym
import torch.nn.functional as F

import custom_env

from model import ActorCriticFF


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
    model = ActorCriticFF((2, 84, 84), env.action_space.n)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(f"{opt.model_path}/a3c_tetris_ff.pt"))
        model.cuda()
    else:
        model.load_state_dict(torch.load(f"{opt.mode_path}/a3c_tetris_ff.pt", map_location=lambda storage, loc: storage))

    model.eval()
    state, info = env.reset()
    done = True

    while True:
        matrix_image = torch.from_numpy(state['matrix_image']).to(model.device)
        falling_shape = torch.from_numpy(state['falling_shape']).to(model.device)
        state = torch.cat((matrix_image, falling_shape), dim=0).to(model.device)
        policy, _ = model(state)
        probs = F.softmax(policy, dim=1)
        action = torch.argmax(probs).item()
        state, reward, done, _, info = env.step(action)
        sleep(0.3)

        if done:
            state, info = env.reset()

if __name__ == "__main__":
    opt = get_args()
    test(opt)