import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import UNREAL
from PIL import Image as im
from utils import make_env, preprocessing
from moviepy.video.fx.resize import resize
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

skip = {9: 6, 10: 5, 11: 5, 12: 5, 13: 4, 14: 4, 15: 4, 16: 3, 17: 3, 18: 3, 19: 2}

if __name__ == "__main__":
    agent = UNREAL(
        n_actions=6, n_inputs=(84, 84, 3), hidden_size=256, device=torch.device("cpu")
    )
    checkpoint = torch.load(
        "trained_models/UNREAL-cont.pt",
        weights_only=True,
        map_location=torch.device("cpu"),
    )
    agent.load_state_dict(checkpoint)
    agent.eval()

    observations = []
    action_taken = []
    probs = []
    ep_length = []

    for level in range(11):
        env = make_env(
            id="TetrisA-v3", level=level + 9, resize=84, skip=1, render_mode="rgb_array"
        )

        action = torch.tensor([0]).long()
        obs, info = env.reset(seed=42)
        reward = torch.zeros(1, 1)
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        level_prob = []
        frames = []
        blocks = 0
        eps_l = 0
        acts = []

        for i in range(87):
            obs, reward, done, _, info = env.step(0)

        env.skip = 2

        with torch.no_grad():
            while blocks < 3:
                eps_l += 1
                obs = preprocessing(obs)
                state_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = torch.nn.functional.one_hot(action, num_classes=env.action_space.n)
                reward = torch.tensor([[reward]]).float()
                policy, value, hx, cx = agent(
                    state_tensor, action, reward, (hx, cx)
                )
                frames.append(env.render().copy())
                action = policy.argmax().unsqueeze(0)
                obs, reward, done, _, info = env.step(action.item())
                acts.append(action.item())
                blocks = sum(info["statistics"].values())
                level_prob.append(policy.squeeze().numpy().tolist())

        # frame = im.fromarray(obs.copy())
        # frame.save(f"./UNREAL-eval/pixel/Level {level + 10}.png")
        # policy_str = np.array2string(policy.squeeze().numpy(), separator=', ')
        # observations.append([np.mean(obs / 255.0), policy_str])
        action_taken.append(acts)
        probs.append(level_prob)
        ep_length.append(eps_l)

        video = ImageSequenceClip(
            frames[::4], fps=4,
        ).fx(resize, width=480)
        video.write_videofile(f"./UNREAL-eval/pixel/{level+9}-nofs.mp4")

    data = []
    for i in range(11):
        data.append([i+9, action_taken[i], probs[i], ep_length[i]])
    df = pd.DataFrame(data, columns=["level", "action sequence", "policy", "episode length"])
    print(df)
    df.to_csv("UNREAL-eval/pixel/pixels-nofs.csv")
