import gym_tetris

import time
import numpy as np

from tqdm import tqdm
from torch import device, manual_seed
from gymnasium.wrappers import (
    ResizeObservation,
    GrayscaleObservation,
    NormalizeObservation,
    FrameStackObservation,
    RecordEpisodeStatistics,
)

from torch import Tensor, float32
from wrapper import FrameSkipWrapper, RecordVideo
# from wrapper import FrameSkipWrapper, RecordVideo
from torchvision.transforms import v2
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT
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
    skip: int = 2,
    framestack: int | None = None,
    normalize = False,
    record = False,
    path: str | None = "./videos",
    format: str | None = "gif",
    level: int = 0,
    num_games: int | None = None
):
    
    manual_seed(42)
    np.random.seed(42)
    make_params = {
        "render_mode": "rgb_array" if record else render_mode,
        "level": level
    }

    env = gym_tetris.make(id, **make_params)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = FrameSkipWrapper(env, skip=skip)

    if grayscale:
        env = GrayscaleObservation(env, keep_dim=True)
    if resize:
        env = ResizeObservation(env, (resize, resize))
    if framestack:
        env = FrameStackObservation(env, framestack)
    if normalize:
        env = NormalizeObservation(env)
    if record:
        env = RecordVideo(env, path, format)
        env = RecordEpisodeStatistics(env, buffer_length=num_games)
    # else:
        # env = FrameSkipWrapper(env, 2)

    return env


def ensure_share_grads(
    local_model,
    global_model,
    device: device,
):
    for local_param, global_param in zip(
        local_model.parameters(), global_model.parameters()
    ):
        # for name, param in local_model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} grad norm: {param.grad.norm()}")
        #     else:
        #         print(f"{name} is none")
        if global_param.grad is not None:
            return
        else:
            global_param._grad = local_param.grad.cpu()


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


def pixel_diff(state, new_state, cell_size=4):
    diff = np.abs(new_state[:, 2:-2, 2:-2] - state[:, 2:-2, 2:-2])
    m = np.mean(diff, 0)
    region = m.shape[0] // cell_size, cell_size, m.shape[1] // cell_size, cell_size
    pixel_change = m.reshape(region).mean(-1).mean(1)

    return pixel_change


def batch_pixel_diff(state, new_state, cell_size=4):
    diff = np.abs(new_state[:, :, 2:-2, 2:-2].cpu() - state[:, :, 2:-2, 2:-2].cpu())
    m = diff.mean(dim=1)
    n_envs, h, w = m.shape
    h_cells, w_cells = h // cell_size, w // cell_size
    reshaped = m.view(n_envs, h_cells, cell_size, w_cells, cell_size)
    pixel_change = reshaped.mean(dim=(-1, -3))  
    return pixel_change