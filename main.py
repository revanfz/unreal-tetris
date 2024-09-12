import os

os.environ["OMP_NUM_THREADS"] = "1"

import json
import torch
import shutil
import argparse
import torch.multiprocessing as mp

from model import UNREAL
from worker import worker
from optimizer import SharedAdam, SharedRMSprop
from torch import manual_seed, load
from gym_tetris.actions import MOVEMENT
from utils import update_progress
from torch.cuda import manual_seed as cuda_manual_seed


def get_args():
    parser = argparse.ArgumentParser(
        """
            Implementation model A3C: 
            IMPLEMENTASI ALGORITMA ASYNCHRONOUS ADVANTAGE ACTOR-CRITIC (A3C)
            UNTUK MENGHASILKAN AGEN CERDAS (STUDI KASUS: PERMAINAN TETRIS)
        """
    )
    parser.add_argument("--lr", type=float, default=0.00058, help="Learning rate")
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount factor for rewards"
    )
    parser.add_argument("--beta", type=float, default=0.0006, help="entropy coefficient")
    parser.add_argument("--task-weight", type=float, default=0.02426, help="task weight")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="rmsprop",
        help="optimizer yang digunakan",
    )
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
        help="jumlah episode sebelum menyimpan checkpoint model",
    )
    parser.add_argument(
        "--max-steps", type=int, default=2e7, help="Maksimal step pelatihan"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=256, help="Jumlah hidden size"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=mp.cpu_count(),
        help="Jumlah agen yang berjalan secara asinkron",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="tensorboard",
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
        device = torch.device("cpu")
        manual_seed(42)

        if not os.path.isdir(params.log_path):
            # shutil.rmtree(params.log_path)
            os.makedirs(params.log_path)

        global_model = UNREAL(
            (3, 84, 84),
            len(MOVEMENT),
            device=torch.device("cpu"),
            hidden_size=params.hidden_size,
            beta=params.beta,
            gamma=params.gamma
        )

        if opt.optimizer == "adam":
            optimizer = SharedAdam(global_model.parameters(), lr=params.lr)
        elif opt.optimizer == "rmsprop":
            optimizer = SharedRMSprop(global_model.parameters(), lr=params.lr)

        processes = []
        global_steps = mp.Value("i", 0)
        global_episodes = mp.Value("i", 0)
        global_rewards = mp.Value("f", 0.0)
        res_queue = mp.Queue()
        manager = mp.Manager()
        shared_dict = manager.dict()

        if opt.resume_training:
            if os.path.isdir(opt.model_path):
                file_ = f"{opt.model_path}/a3c_checkpoint.tar"
                if os.path.isfile(file_):
                    checkpoint = load(file_, weights_only=True)
                    global_model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    global_steps = mp.Value("i", checkpoint["steps"])
                    global_episodes = mp.Value("i", checkpoint["episodes"])
                    with open("checkpoint.json") as f:
                        agent_checkpoint = json.load(f)
                    print(
                        f"Resuming training for previous model, state: Steps {checkpoint['steps']}, Episodes: {checkpoint['episodes']}\nAgent state: {agent_checkpoint}"
                    )
                else:
                    print("File checkpoint belum ada, memulai training...")
            else:
                print("Membuat direktori model...")
                os.makedirs(opt.model_path)
                print("Memulai training...")

        global_model.train()
        optimizer.share_memory()

        progress_process = mp.Process(
            target=update_progress, args=(global_steps, opt.max_steps)
        )
        progress_process.start()
        processes.append(progress_process)

        for rank in range(params.num_agents):
            process = mp.Process(
                target=worker,
                args=(
                    rank,
                    global_model,
                    optimizer,
                    global_steps,
                    global_episodes,
                    shared_dict,
                    params,
                    device,
                    (
                        agent_checkpoint[f"agent_{rank}"]
                        if opt.resume_training
                        and os.path.isfile(f"{opt.model_path}/a3c_checkpoint.tar")
                        else 0
                    ),
                    (
                        checkpoint["num_tries"]
                        if opt.resume_training and os.path.isfile(f"{opt.model_path}/a3c_checkpoint.tar")
                        else 1
                    ),
                ),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

    except (KeyboardInterrupt, mp.ProcessError) as e:
        print("Multiprocessing dihentikan...")
        raise KeyboardInterrupt(f"Program dihentikan")
    
    except Exception as e:
        raise Exception(f"{e}")

    finally:
        for process in processes:
            process.terminate()
        with open("checkpoint.json", "w") as f:
            json.dump(dict(shared_dict), f, indent=4)


if __name__ == "__main__":
    try:
        done = True
        mp.set_start_method("spawn")
        opt = get_args()
        train(opt)

    except KeyboardInterrupt as e:
        done = False
        print("Program dihentikan...")

    except Exception as e:
        done = False
        print(f"Error:\t{e} :X")

    finally:
        print(f"Pelatihan {'selesai' if done else 'gagal'}.")
