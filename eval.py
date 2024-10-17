import os
import torch
import argparse
import pandas as pd
import torch.nn.functional as F

from pprint import pp
from model import UNREAL
from utils import make_env, preprocessing


def get_args():
    parser = argparse.ArgumentParser(
        """
            Evaluasi model A3C: 
            IMPLEMENTASI ALGORITMA ASYNCHRONOUS ADVANTAGE ACTOR-CRITIC (A3C)
            UNTUK MENGHASILKAN AGEN CERDAS (STUDI KASUS: PERMAINAN TETRIS)
        """
    )
    parser.add_argument("--lr", type=float, default=0.00029, help="Learning rate")
    parser.add_argument(
        "--gamma", type=float, default=0.94257, help="discount factor for rewards"
    )
    parser.add_argument(
        "--beta", type=float, default=0.00067, help="entropy coefficient"
    )
    parser.add_argument(
        "--task-weight", type=float, default=0.09855, help="task weight"
    )
    parser.add_argument(
        "--test-case", type=int, default=1, help="Nomor test case"
    )
    parser.add_argument(
        "--num-tries", type=int, default=5, help="Jumlah permainan untuk dievaluasi"
    )
    args = parser.parse_args()
    return args

params = get_args()
data_dir = "./tetris-agent/csv"
video_path = f"./tetris-agent/videos/{params.test_case}"

if __name__ == "__main__":
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    
    if not os.path.isdir(video_path):
        os.makedirs(video_path, exist_ok=True)
    if os.listdir(video_path):
        raise Exception("Folder is not empty. Folder berisi hasil runs sebelumnya")

    env = make_env(
        record=True,
        resize=84,
        path=video_path,
        level = params.test_case - 1
    )
    device = torch.device("cpu")

    model =  UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=env.action_space.n,
        hidden_size=256,
        device=device,
        beta=params.beta,
        gamma=params.gamma,
    )
    model.load_state_dict(
        torch.load(
            "trained_models/a3c_tetris.pt",
            weights_only=True,
        ),
    )
    model.eval()

    episode = 1
    done = True

    data = {
        "lines": [],
        "score": [],
        "rewards": [],
        "block_placed": [],
        "episode_time": [],
        "episode_length": [],
    }
    
    # while True:
    while episode <= params.num_tries:
        if done:
            state, info = env.reset()
            state = preprocessing(state)
            prev_action = torch.zeros(1, env.action_space.n).to(device)
            prev_reward = torch.zeros(1, 1).to(device)
            hx = torch.zeros(1, 256).to(device)
            cx = torch.zeros(1, 256).to(device)
        else:
            hx = hx.data
            cx = cx.data

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            policy, _, _, hx, cx = model(
                state_tensor, prev_action, prev_reward, (hx, cx)
            )

            action = policy.cpu().argmax().unsqueeze(0)

            next_state, reward, done, _, info = env.step(action.item(), info)
            next_state = preprocessing(next_state)
            prev_action = F.one_hot(action, num_classes=env.action_space.n).to(
                device
            )
            prev_reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
            state = next_state

        if done:
            episode += 1
            data["lines"].append(info['number_of_lines'])
            data["block_placed"].append(sum(info["statistics"].values()))
            data["rewards"].append(info['episode']['r'][0])
            data["episode_length"].append(info['episode']['l'][0])
            data["episode_time"].append(info['episode']['t'][0])
            data["score"].append(info['score'])
        
    df = pd.DataFrame(data)
    pp(df)
    df.to_csv(f"{data_dir}/{params.test_case}.csv", index=False)