import os
os.environ["OMP_NUM_THREADS"] = "1"

import shutil
import sys
import torch
import argparse
import torch.multiprocessing as mp

from src.model import ActorCriticNetwork
from src import worker, SharedAdam, update_progress, make_env


def get_args():
    parser = argparse.ArgumentParser(
        """
            Implementation model A3C: 
            IMPLEMENTASI ALGORITMA ASYNCHRONOUS ADVANTAGE ACTOR-CRITIC (A3C)
            UNTUK MENGHASILKAN AGEN CERDAS (STUDI KASUS: PERMAINAN TETRIS)
        """
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount factor for rewards"
    )
    parser.add_argument("--beta", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=5,
        help="jumlah step sebelum mengupdate parameter global",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=5e3,
        help="jumlah steps sebelum menyimpan model",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1e6, help="Maksimal step pelatihan"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=512, help="Jumlah hidden size"
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
        "--use-cuda",
        type=bool,
        default=True,
        help="Mode render dari environment",
    )
    parser.add_argument(
        "--resume-training",
        type=bool,
        default=False,
        help="Load weight from previous trained stage",
    )
    args = parser.parse_args()
    return args


def train(opt: dict) -> None:
    try:
        device = torch.device("cpu")
        env = make_env()

        global_model = ActorCriticNetwork(
            input_channels=env.observation_space.shape[0],
            num_actions=env.action_space.n,
            hidden_size=opt.hidden_size,
            device=torch.device("cpu"),
        )
        global_model.share_memory()
        optimizer = SharedAdam(global_model.parameters(), lr=opt.lr)

        processes = []
        global_episodes = mp.Value("i", 0)
        global_steps = mp.Value("i", 0)

        if opt.resume_training:
            if os.path.isdir(opt.model_path):
                file_ = f"{opt.model_path}/a3c_checkpoint.tar"
                if os.path.isfile(file_):
                    checkpoint = torch.load(file_, weights_only=True)
                    global_model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    global_steps = mp.Value("i", checkpoint["steps"])
                    global_episodes = mp.Value("i", checkpoint["episodes"])
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
            shutil.rmtree(opt.log_path)

        progress_process = mp.Process(
            target=update_progress, args=(global_steps, opt.max_steps)
        )
        progress_process.start()
        processes.append(progress_process)

        for rank in range(opt.num_agents):
            process = mp.Process(
                target=worker,
                args=(
                    rank,
                    global_model,
                    optimizer,
                    global_steps,
                    global_episodes,
                    opt,
                    device
                ),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join(timeout=10)

    except (KeyboardInterrupt, mp.ProcessError, Exception) as e:
        print("Multiprocessing dihentikan...")
        raise Exception(f"{e}")
    

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
        opt = get_args()
        train(opt)

    except (KeyboardInterrupt, Exception) as e:
        print("Program dihentikan...")

    finally:
        print("Pelatihan selesai.")
        sys.exit()