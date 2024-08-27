import time
import logging
import numpy as np

from tqdm import tqdm
from torch import device
from model import UNREAL
from torchvision.transforms import v2
from torch.multiprocessing import Queue
from torch import Tensor, float32, stack
from torch.utils.tensorboard import SummaryWriter
from multiprocessing.sharedctypes import Synchronized


def preprocessing(state: np.ndarray) -> Tensor:
    preprocess = v2.Compose([v2.ToImage(), v2.ToDtype(float32, scale=True)])
    obs = v2.functional.crop(preprocess(state), 48, 96, 160, 80)
    obs = v2.functional.resize(obs, (84, 84))
    return obs.squeeze(0)


def preprocess_frame_stack(frame_stack: np.ndarray) -> Tensor:
    processed_frames = []
    frame_stack = np.transpose(frame_stack, (2, 0, 1))
    for frame in frame_stack:
        processed_frame = preprocessing(frame)
        processed_frames.append(processed_frame)
    return stack(processed_frames).numpy()


def clip_img(img, size):
    h_margin = int((img.shape[1] - size) / 2)
    v_margin = int((img.shape[2] - size) / 2)
    return img[:, h_margin: -h_margin, v_margin: -v_margin]


def calculate_batch_mean(img, batch_size):
    mean = []
    for i in range(int(img.shape[1] / batch_size)):
        mean.append([])
        for j in range(int(img.shape[2] / batch_size)):
            batch = img[:, i * batch_size: (i + 1) * batch_size, j * batch_size: (j + 1) * batch_size]
            mean[-1].append(np.mean(batch))
    return np.array(mean)


def calculate_batch_reward(observation, next_observation, batch_size=4):
    assert observation.shape[1] % batch_size == 0 and observation.shape[2] % batch_size == 0
    observation_mean = calculate_batch_mean(observation, batch_size)
    next_observation_mean = calculate_batch_mean(next_observation, batch_size)
    absolute_difference = np.abs(observation_mean - next_observation_mean)
    return absolute_difference


def ensure_share_grads(
    local_model: UNREAL,
    global_model: UNREAL,
    device: device,
):
    for local_param, global_param in zip(
        local_model.parameters(), global_model.parameters()
    ):
        if global_param.grad is None:
            # if local_param.grad is not None:
            if device.type == "cuda":
                global_param._grad = local_param.grad.cpu()
            else:
                global_param._grad = local_param.grad


def model_logger(model_queue: Queue, path: str) -> None:
    writer = SummaryWriter(path)
    while True:
        while not model_queue.empty():
            log_data = model_queue.get()
            if log_data is None:
                writer.close()
                return

            episode = log_data["episode"]
            writer.add_scalar(f"Global/Rewards", log_data["mean_reward"], episode)

            writer.add_scalar(f"Global/Entropy", log_data["entropy"], episode)

            if episode % 100 == 0:
                writer.flush()

            if episode % 500 == 0:
                print("Now in steps {}.".format(log_data["steps"]))


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s : %(message)s")
    fileHandler = logging.FileHandler(log_file, mode="a")
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


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
