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
total_test_case = 10
data_dir = "./UNREAL-eval/transfer/csv"

if __name__ == "__main__":
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    device = torch.device("cpu")
    checkpoint = torch.load("trained_models/UNREAL-cont-fine-tuning.pt", weights_only=True, map_location=torch.device("cpu"))

    model = UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=6,
        hidden_size=256,
        device=device,
    )
    model.load_state_dict(checkpoint)
    model.eval()
    for test_case in range(params.start_case - 1, total_test_case):
        video_path = f"./UNREAL-eval/transfer/videos/{test_case+1}"
        data_path = f"{data_dir}"

        if not os.path.isdir(video_path):
            os.makedirs(video_path, exist_ok=True)
        if not os.path.isdir(data_path):
            os.makedirs(data_path, exist_ok=True)

        env = make_env(
            record=True,
            resize=84,
            path=video_path,
            level=10 + test_case,
            num_games=params.num_tries,
            id="TetrisA-v3",
            render_mode="rgb_array",
            log_every=1,
            skip=2,
            record_statistics=True, 
        )

        data = {
            "lines": [],
            "score": [],
            "rewards": [],
            "block_placed": [],
            "episode_time": [],
            "episode_length": [],
            "action_taken": [],
            "lines_history": [],
            "board_history": [],
        }

        done = True

        for tries in range(params.num_tries):
            while True:
                if done:
                    state, info = env.reset()
                    ep_r = 0
                    state = preprocessing(state)
                    action = F.one_hot(
                        torch.LongTensor([0]), num_classes=env.action_space.n
                    ).to(device)
                    reward = torch.zeros(1, 1).to(device)
                    hx = torch.zeros(1, 256).to(device)
                    cx = torch.zeros(1, 256).to(device)
                    action_taken = list()
                    board_history = []
                    blocks = 1

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                policy, _, hx, cx = model(state_tensor, action, reward, (hx, cx))
                action = policy.argmax().unsqueeze(0)
                action_taken.append(action.item())

                next_state, reward, done, _, info = env.step(action.item())

                if sum(info["statistics"].values()) > blocks:
                    board = env.unwrapped._board
                    board[board == 239] = 0
                    board[board > 0] = 1
                    board_history.append(board.tolist())
                    blocks = sum(info["statistics"].values())

                ep_r += reward
                next_state = preprocessing(next_state)
                action = F.one_hot(action, num_classes=env.action_space.n).to(
                    device
                )
                reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
                state = next_state

                if done:
                    data["lines"].append(info["number_of_lines"])
                    data["block_placed"].append(sum(info["statistics"].values()))
                    data["score"].append(info["score"])
                    data["rewards"].append(ep_r)
                    data["action_taken"].append(action_taken)
                    data["lines_history"].append(env.lines_history.copy())
                    data["board_history"].append(board_history)
                    break

        data["episode_length"] = np.array(env.env.length_queue)
        data["episode_time"] = np.array(env.env.time_queue)
        env.close()

        df = pd.DataFrame(data)
        df.to_csv(f"{data_path}/{test_case+1}.csv", index=False)
        pp(df)
