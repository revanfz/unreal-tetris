import gym_tetris

import time
import numpy as np

from tqdm import tqdm
from gymnasium.wrappers import (
    ResizeObservation,
    GrayscaleObservation,
    FrameStackObservation,
    RecordEpisodeStatistics,
)
from torch import Tensor, float32
from torchvision.transforms import v2
from gym_tetris.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrapper import FrameSkipWrapper, RecordVideo
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
    skip: int = 4,
    framestack: int | None = None,
    record=False,
    path: str | None = "./videos",
    format: str | None = "mp4",
    level: int = 0,
    log_every: int = 1000,
    episode: int = 0,
    num_games: int | None = None,
    record_statistics = False,
):
    make_params = {
        "render_mode": render_mode,
        "level": level,
    }

    env = gym_tetris.make(id, **make_params)
    env.metadata["fps"] = 60
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    if grayscale:
        env = GrayscaleObservation(env, keep_dim=True)
    if resize:
        env = ResizeObservation(env, (resize, resize))
    if framestack:
        env = FrameStackObservation(env, framestack)
    if record:
        # env = RecordVideo(
        #     env,
        #     video_folder="./videos",
        #     episode_trigger=lambda x: x % 250 == True,
        #     name_prefix="tetris",
        #     disable_logger=True,
        # )
        env = RecordVideo(env, path, format, log_every=log_every, episode=episode)
        record_statistics = True
    if record_statistics:
        env = RecordEpisodeStatistics(env, buffer_length=num_games)

    env = FrameSkipWrapper(env, skip=skip, level=level)

    return env


def ensure_share_grads(
    local_model,
    global_model,
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


def update_progress(
    global_steps: Synchronized,
    max_steps: float,
    checkpoint_steps=0,
    desc=None,
    unit=None,
):
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
