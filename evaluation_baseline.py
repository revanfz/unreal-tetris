import os
import torch
import numpy as np
import pandas as pd

from pprint import pp
from model import UNREAL
from utils import make_env, preprocessing
from torch.distributions import Categorical


baseline_agent = UNREAL(
    n_actions=6, n_inputs=(84, 84, 3), hidden_size=256, device=torch.device("cpu")
)
trained_agent = UNREAL(
    n_actions=6, n_inputs=(84, 84, 3), hidden_size=256, device=torch.device("cpu")
)
checkpoint = torch.load(
    "trained_models/UNREAL-tetris.pt",
    weights_only=True,
    map_location=torch.device("cpu"),
)
trained_agent.load_state_dict(checkpoint)
trained_agent.eval()

models = {"baseline": baseline_agent, "trained": trained_agent}
scenarios = ["same-seed", "different-seed"]
N_TEST = 30
data_dir = "./UNREAL-eval/baseline/csv"
video_dir = "./UNREAL-eval/baseline/videos"

if __name__ == "__main__":
    for scenario in scenarios:
        data_path = f"{data_dir}/{scenario}"
        if not os.path.isdir(data_path):
            os.makedirs(data_path)

        for agent in models.keys():
            video_path = f"{video_dir}/{scenario}/{agent}"
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            env = make_env(
                record=True,
                resize=84,
                path=video_path,
                level=19,
                num_games=N_TEST,
                id="TetrisA-v3",
                render_mode="human",
                log_every=1,
                skip=2,
            )
            data = {
                "lines": [],
                "score": [],
                "rewards": [],
                "block_placed": [],
                "episode_time": [],
                "episode_length": [],
                "action_taken": [],
                "line_history": [],
            }
            done = True
            for tries in range(N_TEST):
                while True:
                    if done:
                        if scenario == "same-seed":
                            state, info = env.reset(seed=42)
                        else:
                            state, info = env.reset()
                        ep_r = 0
                        state = preprocessing(state)
                        action = torch.nn.functional.one_hot(
                            torch.LongTensor([0]), num_classes=env.action_space.n
                        )
                        reward = torch.zeros(1, 1)
                        hx = torch.zeros(1, 256)
                        cx = torch.zeros(1, 256)
                        action_taken = list()

                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    policy, _, hx, cx = models[agent](
                        state_tensor, action, reward, (hx, cx)
                    )

                    if agent == "baseline":
                        dist = Categorical(probs=policy)
                        action = dist.sample()
                    else:
                        action = policy.argmax().unsqueeze(0)
                    action_taken.append(action.item())

                    next_state, reward, done, _, info = env.step(action.item())
                    ep_r += reward
                    next_state = preprocessing(next_state)
                    action = torch.nn.functional.one_hot(
                        action, num_classes=env.action_space.n
                    )
                    reward = torch.FloatTensor([reward]).unsqueeze(0)
                    state = next_state

                    if done:
                        data["lines"].append(info["number_of_lines"])
                        data["block_placed"].append(sum(info["statistics"].values()))
                        data["score"].append(info["score"])
                        data["rewards"].append(ep_r)
                        data["action_taken"].append(action_taken)
                        data["line_history"].append(env.lines_history.copy())
                        break

            data["episode_length"] = np.array(env.env.length_queue)
            data["episode_time"] = np.array(env.env.time_queue)
            env.close()

            df = pd.DataFrame(data)
            pp(df)
            df.to_csv(f"{data_path}/{agent}.csv", index=False)
