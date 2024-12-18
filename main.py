import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
import torch
import argparse
import numpy as np
import torch.multiprocessing as mp

from pprint import pp
from model import UNREAL
from worker import worker
from utils import make_env, update_progress
from optimizer import SharedRMSprop, SharedAdam


def get_args():
    parser = argparse.ArgumentParser(
        """
            Implementation model UNREAL: 
            IMPLEMENTASI ARSITEKTUR UNSUPERVISED REINFORCEMENT WITH AUXILIARY LEARNING (UNREAL)
            UNTUK MENGHASILKAN AGEN CERDAS (STUDI KASUS: PERMAINAN TETRIS)
        """
    )
    parser.add_argument("--lr", type=float, default=0.00012, help="Learning rate")
    parser.add_argument(
        "--gamma", type=float, default=0.95, help="discount factor for rewards"
    )
    parser.add_argument(
        "--beta", type=float, default=0.00318, help="entropy coefficient"
    )
    parser.add_argument("--pc-weight", type=float, default=0.05478, help="task weight")
    parser.add_argument("--grad-norm", type=float, default=40, help="Gradient norm clipping")
    parser.add_argument(
        "--unroll-steps",
        type=int,
        default=20,
        help="jumlah step sebelum mengupdate parameter global",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5e3,
        help="jumlah episode sebelum menyimpan checkpoint model",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1e7, help="Maksimal step pelatihan"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=256, help="Jumlah hidden size"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="rmsprop",
        help="optimizer yang digunakan",
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
        torch.manual_seed(42)
        np.random.seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)

        env = make_env(grayscale=False, framestack=None, resize=84)

        global_model = UNREAL(
            n_inputs=(84, 84, 3),
            n_actions=env.action_space.n,
            device=torch.device("cpu"),
            hidden_size=params.hidden_size,
            beta=params.beta,
            gamma=params.gamma,
        )
        global_model.share_memory()
        global_model.train()
        
        if opt.optimizer == "adam":
            optimizer = SharedAdam(global_model.parameters(), lr=params.lr)
        elif opt.optimizer == "rmsprop":
            optimizer = SharedRMSprop(global_model.parameters(), lr=params.lr)

        processes = []
        global_steps = mp.Value("i", 0)
        global_episodes = mp.Value("i", 0)
        global_lines = mp.Value("i", 0)

        load_model = False

        if opt.resume_training:
            if os.path.isdir(opt.model_path):
                file_ = f"{opt.model_path}/UNREAL_checkpoint.tar"
                if os.path.isfile(file_):
                    load_model = True
                    checkpoint = torch.load(file_, weights_only=True)
                    global_model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    global_steps = mp.Value("i", checkpoint["steps"])
                    global_episodes = mp.Value("i", checkpoint["episodes"])
                    global_lines = mp.Value("i", checkpoint["lines"])
                    print(
                        f"Resuming training for previous model, state: Steps {checkpoint['steps']}, Episodes: {checkpoint['episodes']}"
                    )
                else:
                    print("File checkpoint belum ada, memulai training...")
            else:
                print("Membuat direktori model...")
                os.makedirs(opt.model_path)
                print("Memulai training...")

        pp(opt)
        progress_process = mp.Process(
            target=update_progress,
            args=(
                global_steps,
                (opt.max_steps - global_steps.value if load_model else opt.max_steps),
                checkpoint["steps"] if load_model else 0,
            ),
            kwargs=(
                {
                    "desc": (
                        "Resuming Training. Total Steps"
                        if load_model
                        else "Total Steps"
                    ),
                    "unit": "steps"
                }
            ),
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
                    global_lines,
                    params,
                    device,
                ),
            )
            process.start()
            processes.append(process)

        # with tqdm(
        #     total=opt.max_steps - global_steps.value if load_model else opt.max_steps,
        #     desc=f"{'Resuming ' if load_model else ''}Training model",
        #     unit="steps",
        # ) as pbar:
        #     while any(p.is_alive() for p in processes):
        #         with global_steps.get_lock():
        #             pbar.n = global_steps.value - (
        #                 checkpoint["steps"] if load_model else 0
        #             )
        #             pbar.refresh()
        #         time.sleep(0.1)

        for process in processes:
            process.join()

    except (KeyboardInterrupt, mp.ProcessError) as e:
        print("Multiprocessing dihentikan...")
        raise KeyboardInterrupt(f"Program dihentikan")

    finally:
        for process in processes:
            process.terminate()


if __name__ == "__main__":
    try:
        done = False
        mp.set_start_method("spawn")
        opt = get_args()
        train(opt)
        done = True

    except KeyboardInterrupt as e:
        print("Program dihentikan...")

    finally:
        print(f"Pelatihan {'selesai' if done else 'gagal'}.")
