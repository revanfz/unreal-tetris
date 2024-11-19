import os

os.environ["OMP_NUM_THREADS"] = "1"

import torch
import shutil
import argparse
import torch.multiprocessing as mp

from pprint import pp
from model import UNREAL
from worker import worker
from torch import manual_seed, load
from utils import make_env, update_progress
from optimizer import SharedAdam, SharedRMSprop


def get_args():
    parser = argparse.ArgumentParser(
        """
            Implementation model UNREAL: 
            IMPLEMENTASI ARSITEKTUR UNSUPERVISED REINFORCEMENT WITH AUXILIARY LEARNING (UNREAL)
            UNTUK MENGHASILKAN AGEN CERDAS (STUDI KASUS: PERMAINAN TETRIS)
        """
    )
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount factor for rewards"
    )
    parser.add_argument(
        "--beta", type=float, default=0.00102, help="entropy coefficient"
    )
    parser.add_argument(
        "--pc-weight", type=float, default=0.08928, help="task weight"
    )
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
        default=8e3,
        help="jumlah episode sebelum menyimpan checkpoint model",
    )
    parser.add_argument(
        "--max-steps", type=int, default=5e7, help="Maksimal step pelatihan"
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
        default=False,
        help="Load weight from previous trained stage",
    )
    args = parser.parse_args()
    return args


def train(params: argparse.Namespace) -> None:
    try:
        device = torch.device("cpu")
        manual_seed(42)

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

        if opt.optimizer == "adam":
            optimizer = SharedAdam(global_model.parameters(), lr=params.lr)
        elif opt.optimizer == "rmsprop":
            optimizer = SharedRMSprop(global_model.parameters(), lr=params.lr)

        processes = []
        global_steps = mp.Value("i", 0)
        global_episodes = mp.Value("i", 0)
        global_lines = mp.Value("i", 0)
        global_scores = mp.Value("i", 0)

        if opt.resume_training:
            if os.path.isdir(opt.model_path):
                load_model = True
                file_ = f"{opt.model_path}/final.tar"
                if os.path.isfile(file_):
                    checkpoint = load(file_, weights_only=True)
                    global_model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    global_steps = mp.Value("i", checkpoint["steps"])
                    global_episodes = mp.Value("i", checkpoint["episodes"])
                    global_lines = mp.Value("i", checkpoint["lines"])
                    global_scores = mp.Value("i", checkpoint["scores"])
                    print(
                        f"Resuming training for previous model, state: Steps {checkpoint['steps']}, Episodes: {checkpoint['episodes']}"
                    )
                else:
                    print("File checkpoint belum ada, memulai training...")
            else:
                print("Membuat direktori model...")
                os.makedirs(opt.model_path)
                print("Memulai training...")
        else:
            load_model = False


        global_model.train()
        optimizer.share_memory()
        pp(opt)
        progress_process = mp.Process(
            target=update_progress,
            args=(
                global_steps,
                (
                    opt.max_steps - global_steps.value
                    if load_model
                    else opt.max_steps
                ),
                checkpoint["steps"] if load_model else 0
            ),
            kwargs=({"desc": "Resuming Training. Total Steps" if load_model else "Total Steps"})
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
                    global_scores,
                    params,
                    device,
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
