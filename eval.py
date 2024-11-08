import os
import torch
import argparse
import numpy as np
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
        "--num-tries", type=int, default=1, help="Jumlah permainan untuk dievaluasi"
    )
    args = parser.parse_args()
    return args

params = get_args()
total_test_case = 10
data_dir = "./tetris-agent/csv"

if __name__ == "__main__":
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    env = make_env(
        resize=84,
        level=19
    )
    device = torch.device("cpu")
    checkpoint = torch.load("trained_models/unreal_checkpoint.tar", weights_only=True)

    model =  UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=env.action_space.n,
        hidden_size=256,
        device=device,
        beta=params.beta,
        gamma=params.gamma,
    )
    model.load_state_dict(
        # torch.load(
        #     # "trained_models/a3c_tetris.pt",
        #     "trained_models/unreal_checkpoint.tar",
        #     weights_only=True,
        # ),
        checkpoint["model_state_dict"]
    )
    model.eval()
    del env

    test_case = params.test_case
    # while True:
    while test_case <= total_test_case:
        # video_path = f"./tetris-agent/videos/{test_case}"
    
        # if not os.path.isdir(video_path):
        #     os.makedirs(video_path, exist_ok=True)
        # if os.listdir(video_path):
        #     raise Exception("Folder is not empty. Folder berisi hasil runs sebelumnya")
        
        env = make_env(
            # record=True,
            resize=84,
            # path=video_path,
            level = test_case,
            num_games = params.num_tries,
            render_mode="human",
        )

        data = {
            "lines": [],
            "score": [],
            "rewards": [],
            "block_placed": [],
            "episode_time": [],
            "episode_length": [],
        }
            
        episode = 1
        done = True

        while episode <= params.num_tries:
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

            action = policy.cpu().argmax().unsqueeze(0)
            print(action.item())

            next_state, reward, done, _, info = env.step(action.item())
            next_state = preprocessing(next_state)
            action = F.one_hot(action, num_classes=env.action_space.n).to(
                device
            )
            reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
            state = next_state

            if done:
                episode += 1
                data["lines"].append(info['number_of_lines'])
                data["block_placed"].append(sum(info["statistics"].values()))
                data["score"].append(info['score'])
                # data["rewards"].append(info["episode"]["r"] + 10 * info["number_of_lines"] - 5)
            
        
        # data["episode_length"] = np.array(env.length_queue)
        # data["episode_time"] = np.array(env.time_queue)
        # del env

        # df = pd.DataFrame(data)
        # pp(df)
        # df.to_csv(f"{data_dir}/{test_case}.csv", index=False)
        # test_case += 1
