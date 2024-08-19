import os

os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import json
import shutil
import argparse
import torch.multiprocessing as mp

from test import test_model
from train import train_model_lstm
from model import ActorCriticLSTM
from optimizer import SharedAdam
from torch import manual_seed, load
from gym_tetris.actions import MOVEMENT
from torch.cuda import manual_seed as cuda_manual_seed
from utils import model_logger, update_progress


def get_args():
    parser = argparse.ArgumentParser(
        """
            Implementation model A3C: 
            IMPLEMENTASI ALGORITMA ASYNCHRONOUS ADVANTAGE ACTOR-CRITIC (A3C)
            UNTUK MENGHASILKAN AGEN CERDAS (STUDI KASUS: PERMAINAN TETRIS)
        """
    )
    parser.add_argument("--lr", type=float, default=7e-4, help="Learning rate")
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount factor for rewards"
    )
    parser.add_argument("--beta", type=float, default=0.5, help="entropy coefficient")
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=5,
        help="jumlah step sebelum mengupdate parameter global",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5e1,
        help="jumlah steps sebelum menyimpan model",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1e3, help="Maksimal step pelatihan"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=256, help="Jumlah hidden size"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=2,
        help="Jumlah agen yang berjalan secara asinkron",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="tensorboard/baseline_a3c_tetris",
        # default="tensorboard/tuned_a3c_tetris",
        help="direktori plotting tensorboard",
    )
    parser.add_argument(
        "--log-test-path",
        type=str,
        default="tensorboard/run_baseline_a3c_tetris",
        # default="tensorboard/tuned_a3c_tetris",
        help="direktori plotting testing tensorboard",
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
        manual_seed(42)
        if opt.use_cuda:
            cuda_manual_seed(42)

        if os.path.isdir(opt.log_path):
            shutil.rmtree(opt.log_path)
        os.makedirs(opt.log_path)

        global_model = ActorCriticLSTM((4, 84, 84), len(MOVEMENT), opt.hidden_size)
        global_model.share_memory()

        optimizer = SharedAdam(global_model.parameters(), lr=opt.lr)
        optimizer.share_memory()

        processes = []
        global_episodes = mp.Value("i", 0)
        global_steps = mp.Value("i", 0)
        global_rewards = mp.Value("f", 0.0)
        start_time = time.time()
        max_train_time = 14.5 * 3600
        stop_event = mp.Event()
        res_queue = mp.Queue()
        manager = mp.Manager()
        shared_dict = manager.dict()

        num_tests = 0
        if opt.resume_training:
            if os.path.isdir(opt.model_path):
                file_ = f"{opt.model_path}/baseline_a3c_tetris.tar"
                # file_ = f"{opt.model_path}/tuned_a3c_tetris.tar"
                if os.path.isfile(file_):
                    checkpoint = load(file_)
                    global_model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    global_steps = mp.Value("i", checkpoint["steps"])
                    num_tests = checkpoint["num_tests"]
                    with open("checkpoint.json") as f:
                        agent_checkpoint = json.load(f)
                    print(
                        f"Resuming training for previous model, state: Steps {checkpoint['steps']}, Tests: {checkpoint['num_tests']}\nAgent state: {agent_checkpoint}"
                    )
                else:
                    print("File checkpoint belum ada, memulai training...")
            else:
                print("Membuat direktori model...")
                os.makedirs(opt.model_path)
                print("Memulai training...")

        p = mp.Process(
            target=test_model,
            args=(opt, global_model, optimizer, global_steps, num_tests),
        )
        p.start()
        processes.append(p)
        time.sleep(0.01)

        progress_process = mp.Process(
            target=update_progress, args=(global_steps, opt.max_steps)
        )
        progress_process.start()
        processes.append(p)
        time.sleep(0.01)

        for rank in range(opt.num_agents):
            process = mp.Process(
                target=train_model_lstm,
                args=(
                    rank,
                    opt,
                    global_model,
                    optimizer,
                    global_episodes,
                    global_steps,
                    global_rewards,
                    stop_event,
                    start_time,
                    max_train_time,
                    res_queue,
                    shared_dict,
                    (
                        agent_checkpoint[f"agent_{rank}"]
                        if opt.resume_training
                        and os.path.isfile(f"{opt.model_path}/baseline_a3c_tetris.tar")
                        else 0
                    ),
                ),
            )
            process.start()
            processes.append(process)
            time.sleep(0.001)

        logger_process = mp.Process(target=model_logger, args=(res_queue, opt.log_path))
        logger_process.start()

        for process in processes:
            time.sleep(0.001)
            process.join()
        time.sleep(0.001)
        logger_process.join()

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
