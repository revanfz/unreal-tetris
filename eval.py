import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F

from pprint import pp
from model import UNREAL
from utils import make_env, preprocessing
from torch.distributions import Categorical


def get_args():
    parser = argparse.ArgumentParser(
        """
            Evaluasi model UNREAL: 
            IMPLEMENTASI ARSITEKTUR UNSUPERVISED REINFORCEMENT WITH AUXILIARY LEARNING (UNREAL)
            UNTUK MENGHASILKAN AGEN CERDAS (STUDI KASUS: PERMAINAN TETRIS)
        """
    )
    parser.add_argument(
        "--start-case", type=int, default=1, help="Starting point test case"
    )
    parser.add_argument(
        "--num-tries", type=int, default=30, help="Jumlah permainan untuk dievaluasi"
    )
    args = parser.parse_args()
    return args

params = get_args()
total_test_case = 20
data_dir = "./UNREAL-tetris/csv"

if __name__ == "__main__":
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    device = torch.device("cpu")
    checkpoint = torch.load("trained_models/final.pt", weights_only=True)

    model =  UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=12,
        hidden_size=256,
        device=device,
    )
    model.load_state_dict(
        checkpoint
    )
    model.eval()

    for test_case in range(total_test_case):
        id = params.start_case + test_case
        video_path = f"./UNREAL-tetris/videos/{params.start_case + test_case}"
    
        if not os.path.isdir(video_path):
            os.makedirs(video_path, exist_ok=True)
        if os.listdir(video_path):
            raise Exception("Folder is not empty. Folder berisi hasil runs sebelumnya")
        
        env = make_env(
            record=True,
            resize=84,
            path=video_path,
            level = id - 1,
            num_games = params.num_tries,
            id="TetrisA-v0"
        )

        data = {
            "lines": [],
            "score": [],
            "rewards": [],
            "block_placed": [],
            "episode_time": [],
            "episode_length": [],
        }

        done = True

        for tries in range(params.num_tries):
            while True:
                if done:
                    state, info = env.reset()
                    state = preprocessing(state)
                    action = F.one_hot(torch.LongTensor([0]), num_classes=env.action_space.n).to(device)
                    reward = torch.zeros(1, 1).to(device)
                    hx = torch.zeros(1, 256).to(device)
                    cx = torch.zeros(1, 256).to(device)
                else:
                    hx = hx.detach()
                    cx = cx.detach()

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                policy, _, hx, cx = model(
                    state_tensor, action, reward, (hx, cx)
                )

                dist = Categorical(probs=policy)
                action = dist.sample()

                next_state, reward, done, _, info = env.step(action.item())
                next_state = preprocessing(next_state)
                action = F.one_hot(action, num_classes=env.action_space.n).to(
                    device
                )
                reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
                state = next_state

                if done:
                    data["lines"].append(info['number_of_lines'])
                    data["block_placed"].append(sum(info["statistics"].values()))
                    data["score"].append(info['score'])
                    data["rewards"].append(info["episode"]["r"])
                    break
            
        
        data["episode_length"] = np.array(env.length_queue)
        data["episode_time"] = np.array(env.time_queue)
        del env

        df = pd.DataFrame(data)
        pp(df)
        df.to_csv(f"{data_dir}/{id}.csv", index=False)
