import torch
import argparse
import gymnasium as gym
import torch.nn.functional as F

import custom_env

from model import ActorCriticFF, ActorCriticLSTM


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
    model = ActorCriticLSTM(2, env.action_space.n)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(f"{opt.model_path}/a3c_tetris_lstm.pt"))
        model.cuda()
    else:
        model.load_state_dict(torch.load(f"{opt.mode_path}/a3c_tetris_lstm.pt", map_location=lambda storage, loc: storage))

    model.eval()
    state, info = env.reset()
    
    matrix_image = torch.from_numpy(state['matrix_image']).to(model.device)
    falling_shape = torch.from_numpy(state['falling_shape']).to(model.device)
    state = torch.cat((matrix_image, falling_shape), dim=0).to(model.device)
    if torch.cuda.is_available():
        state = state.cuda()
    done = True

    while True:
        if done:
            hx = torch.zeros((1, 256), dtype=torch.float)
            cx = torch.zeros((1, 256), dtype=torch.float)
        else:
            hx = hx.detach()
            cx = cx.detach()
        if model.device == "cuda":
            hx = hx.cuda()
            cx = cx.cuda()

        policy, value, hx, cx = model(state, hx, cx)
        probs = F.softmax(policy, dim=1)
        action = torch.argmax(probs).item()
        state, reward, done, _, info = env.step(action)
        matrix_image = torch.from_numpy(state['matrix_image']).to(model.device)
        falling_shape = torch.from_numpy(state['falling_shape']).to(model.device)
        state = torch.cat((matrix_image, falling_shape), dim=0).to(model.device)

        if done:
            state, info = env.reset()
            matrix_image = torch.from_numpy(state['matrix_image']).to(model.device)
            falling_shape = torch.from_numpy(state['falling_shape']).to(model.device)
            state = torch.cat((matrix_image, falling_shape), dim=0).to(model.device)
        
        if torch.cuda.is_available():
            state = state.cuda()

if __name__ == "__main__":
    opt = get_args()
    test(opt)