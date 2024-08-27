import os

from replay_buffer import ReplayBuffer

os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import json
import torch
import shutil
import argparse
import torch.multiprocessing as mp

from model import UNREAL
from worker import worker
from optimizer import SharedAdam
from torch import manual_seed, load
from gym_tetris.actions import MOVEMENT
from utils import model_logger, update_progress
from torch.cuda import manual_seed as cuda_manual_seed


def get_args():
    parser = argparse.ArgumentParser(
        """
            Implementation model A3C: 
            IMPLEMENTASI ALGORITMA ASYNCHRONOUS ADVANTAGE ACTOR-CRITIC (A3C)
            UNTUK MENGHASILKAN AGEN CERDAS (STUDI KASUS: PERMAINAN TETRIS)
        """
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--gamma", type=float, default=0.9, help="discount factor for rewards"
    )
    parser.add_argument("--beta", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument(
        "--unroll-steps",
        type=int,
        default=20,
        help="jumlah step sebelum mengupdate parameter global",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1e3,
        help="jumlah steps sebelum menyimpan model",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1e6, help="Maksimal step pelatihan"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=256, help="Jumlah hidden size"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=4,
        help="Jumlah agen yang berjalan secara asinkron",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="tensorboard/a3c_tetris",
        help="direktori plotting tensorboard",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="trained_models",
        help="direktori penyimpanan model hasil training",
    )
    parser.add_argument(
        "--resume-training",
        type=bool,
        default=True,
        help="Load weight from previous trained stage",
    )
    args = parser.parse_args()
    return args


def train(params: argparse.Namespace) -> None:
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            cuda_manual_seed(42)
        else:
            manual_seed(42)

        if os.path.isdir(params.log_path):
            shutil.rmtree(params.log_path)
        os.makedirs(params.log_path)

        global_model = UNREAL((3, 84, 84), len(MOVEMENT), device=torch.device("cpu"), hidden_size=params.hidden_size)
        global_model.train()

        optimizer = SharedAdam(global_model.parameters(), lr=params.lr)
        optimizer.share_memory()

        shared_replay_buffer = ReplayBuffer(2000)

        processes = []
        global_steps = mp.Value("i", 0)
        res_queue = mp.Queue()
        manager = mp.Manager()
        shared_dict = manager.dict()
        
        for rank in range(params.num_agents):
            process = mp.Process(
                target=worker,
                args=(
                   rank,
                   global_model,
                   optimizer,
                   shared_replay_buffer,
                   global_steps,
                   params
                ),
            )
            process.start()
            processes.append(process)
            time.sleep(0.1)

        for process in processes:
            time.sleep(0.1)
            process.join()

    except (KeyboardInterrupt, mp.ProcessError) as e:
        print("Multiprocessing dihentikan...")
        raise KeyboardInterrupt

    finally:
        with open("checkpoint.json", "w") as f:
            json.dump(dict(shared_dict), f, indent=4)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
        opt = get_args()
        train(opt)

    except KeyboardInterrupt as e:
        print("Program dihentikan...")
        sys.exit()

    finally:
        print("Pelatihan selesai.")
