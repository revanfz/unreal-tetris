import gym_tetris

import time
import numpy as np

from tqdm import tqdm
from torch import device
from gymnasium.wrappers import (
    FrameStack,
    ResizeObservation,
    GrayScaleObservation,
    NormalizeObservation,
    RecordEpisodeStatistics
)

from torch import Tensor, float32
from wrapper import FrameSkipWrapper
from torchvision.transforms import v2
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import MOVEMENT
from multiprocessing.sharedctypes import Synchronized


def preprocessing(state: np.ndarray, pixel_control: bool = False) -> Tensor:
    if pixel_control:
        obs = v2.CenterCrop(80)(state)
    else:
        preprocess = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(float32, scale=True),
                # v2.Lambda(lambda x: v2.functional.crop(x, 48, 96, 160, 80)),
                # v2.Resize((84, 84))
            ]
        )
        obs = preprocess(state.copy()).numpy()
    return obs

def make_env(
    id: str = "TetrisA-v3",
    grayscale: bool = False,
    resize: int = 0,
    render_mode="rgb_array",
    framestack: int = 4,
    normalize = False,
    record = False
):
    env = gym_tetris.make(id, render_mode=render_mode)
    env = JoypadSpace(env, MOVEMENT)
    if grayscale:
        env = GrayScaleObservation(env, keep_dim=True)
    if resize:
        env = ResizeObservation(env, resize)
    if framestack:
        env = FrameStack(env, framestack)
    if normalize:
        env = NormalizeObservation(env)
    if record:
        env = RecordEpisodeStatistics(env, deque_size=300)
    env = FrameSkipWrapper(env, 2)

    return env


def ensure_share_grads(
    local_model,
    global_model,
    device: device,
):
    for local_param, global_param in zip(
        local_model.parameters(), global_model.parameters()
    ):
        if global_param.grad is not None:
            return
        if device.type == "cuda":
            global_param._grad = local_param.grad.cpu()
        else:
            global_param._grad = local_param.grad


def update_progress(global_steps: Synchronized, max_steps: float, checkpoint_steps = 0, desc=None, unit=None):
    pbar = tqdm(
        total=max_steps,
        desc="Total Steps" if not desc else desc,
        unit="step" if not unit else unit,
    )
    while global_steps.value - checkpoint_steps < max_steps:
        pbar.n = global_steps.value - checkpoint_steps
        pbar.refresh()
        time.sleep(0.1)
    pbar.close()
