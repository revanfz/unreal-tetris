
from multiprocessing.sharedctypes import Synchronized
import time
import logging
import numpy as np

from tqdm import tqdm
from torchvision.transforms import v2
from torch.multiprocessing import Queue
from torch import Tensor, float32, stack
from torchvision.utils import save_image
from model import ActorCriticFF, ActorCriticLSTM
from torch.utils.tensorboard import SummaryWriter


def preprocessing(state: np.ndarray) -> Tensor:
    preprocess = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(float32, scale=True)
    ])
    obs = v2.functional.crop(preprocess(state), 48, 96, 160, 80)
    obs = v2.functional.resize(obs, (84, 84))
    return obs.squeeze(0)
    # return obs


def preprocess_frame_stack(frame_stack: np.ndarray) -> Tensor:
    processed_frames = []
    for frame in frame_stack:
        processed_frame = preprocessing(frame)
        processed_frames.append(processed_frame)

    return stack(processed_frames)


def ensure_share_grads(local_model: ActorCriticLSTM, global_model: ActorCriticLSTM, device: str):
    for local_param, global_param in zip(
        local_model.parameters(), global_model.parameters()
    ):
        if global_param.grad is not None and device != "cuda":
            return
        if device == 'cpu':
            global_param._grad = local_param.grad
        elif device == 'cuda':
            global_param._grad = local_param.grad.cpu()

    
def model_logger(model_queue: Queue, path: str) -> None:
    writer = SummaryWriter(path)
    while True:
        while not model_queue.empty():
            log_data = model_queue.get()
            if log_data is None:
                writer.close()
                return
            
            episode = log_data["episode"]
            writer.add_scalar(
                f"Global/Rewards",
                log_data["mean_reward"],
                episode
            )

            writer.add_scalar(
                f"Global/Entropy",
                log_data["entropy"],
                episode
            )

            if episode % 100 == 0:
                writer.flush()

            if episode % 500 == 0:
                print("Now in steps {}.".format(log_data["steps"]))
            


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

def update_progress(global_steps: Synchronized, max_steps: float, desc: None, unit: None):
    pbar = tqdm(total=max_steps, desc="Total Steps" if not desc else desc, unit="step" if not unit else unit)
    while global_steps.value < max_steps:
        pbar.n = global_steps.value
        pbar.refresh()
        time.sleep(0.1)
    pbar.close()