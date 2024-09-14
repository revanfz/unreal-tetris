import gym_tetris

import time
import numpy as np

from tqdm import tqdm
from torch import float32
from gym.wrappers import (
    FrameStack,
    NormalizeReward,
    ResizeObservation,
    GrayScaleObservation,
)
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import MOVEMENT
from torchvision.transforms import v2
from multiprocessing.sharedctypes import Synchronized


def preprocessing(obs: np.ndarray):
    return v2.Compose(
        [
            # v2.ToImage(),
            # v2.ToDtype(float32, scale=True),
            v2.Lambda(lambda x: v2.functional.crop(x, 48, 96, 160, 120)),
            v2.Resize((84, 84))
        ]
    )(obs)


def update_progress(global_steps: Synchronized, max_steps: float, desc=None, unit=None):
    pbar = tqdm(
        total=max_steps,
        desc="Total Steps" if not desc else desc,
        unit="step" if not unit else unit,
    )
    while global_steps.value < max_steps:
        pbar.n = global_steps.value
        pbar.refresh()
        time.sleep(0.1)
    pbar.close()


def make_env(
    id: str = "TetrisA-v3",
    grayscale: bool = True,
    resize: int = 84,
    render_mode="rgb_array",
    framestack: int = 4,
):
    env = gym_tetris.make(id, apply_api_compatibility=True, render_mode=render_mode)
    env.metadata["render_modes"] = ["rgb_array", "human"]
    env.metadata["render_fps"] = 60
    env = JoypadSpace(env, MOVEMENT)
    if grayscale:
        env = GrayScaleObservation(env, keep_dim=True)
    if resize:
        env = ResizeObservation(env, resize)
    env = FrameStack(env, framestack)

    return env
