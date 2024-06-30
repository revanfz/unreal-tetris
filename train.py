import os
import sys
import torch
import shutil
import argparse
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

import custom_env

from model import ActorCriticFF
from optimizer import SharedAdam
from process import local_train


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
    parser.add_argument("--beta", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument(
        "--sync-steps",
        type=int,
        default=5,
        help="jumlah step sebelum mengupdate parameter global",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5000,
        help="jumlah step sebelum menyimpan model",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1e6, help="Maksimal step pelatihan"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=16,
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
        "--render-mode",
        type=str,
        default="rgb_array",
        help="Mode render dari environment",
    )
    parser.add_argument(
        "--framestack", type=int, default=4, help="Numebr of framestack to be feed"
    )
    parser.add_argument(
        "--load-model",
        type=bool,
        default=False,
        help="Load weight from previous trained stage",
    )
    args = parser.parse_args()
    return args


def train(opt):
    try:
        torch.manual_seed(42)
        if os.path.isdir(opt.log_path):
            shutil.rmtree(opt.log_path)
        os.makedirs(opt.log_path)

        if not os.path.isdir(opt.model_path):
            os.makedirs(opt.model_path)

        mp.get_context("spawn")
        env = gym.make("SmartTetris-v0")

        global_model = ActorCriticFF(opt.framestack, env.action_space.n)
        if torch.cuda.is_available():
            global_model.cuda()
        global_model.share_memory()

        del env

        optimizer = SharedAdam(global_model.parameters(), lr=opt.lr)
        optimizer.share_memory()

        if opt.load_model:
            file_ = "{}/a3c_tetris.pt".format(opt.model_path)
            if os.path.isfile(file_):
                global_model.load_state_dict(torch.load(file_))

        processes = []
        global_steps = mp.Value("i", 0)
        global_episodes = mp.Value("i", 0)
        global_rewards = mp.Value("f", 0)
        res_queue = mp.Queue()
        for index in range(opt.num_agents):
            if index == 0:
                process = mp.Process(
                    target=local_train,
                    args=(
                        index,
                        opt,
                        global_model,
                        optimizer,
                        global_episodes,
                        global_steps,
                        global_rewards,
                        res_queue,
                        True,
                    ),
                )
            else:
                process = mp.Process(
                    target=local_train,
                    args=(
                        index,
                        opt,
                        global_model,
                        optimizer,
                        global_episodes,
                        global_steps,
                        global_rewards,
                        res_queue,
                    ),
                )
            process.start()
            processes.append(process)

            res = []
            while True:
                r = res_queue.get()
                if r is not None:
                    res.append(r)
                else:
                    break
            for process in processes:
                process.join()

            # plt.plot(res)
            # plt.ylabel("Moving average ep reward")
            # plt.xlabel("Episode")
            # plt.show()
            
    except (KeyboardInterrupt, mp.ProcessError):
        print("Multiprocessing dihentikan...")
        raise KeyboardInterrupt


if __name__ == "__main__":
    try:
        opt = get_args()
        train(opt)
    except KeyboardInterrupt:
        print("Program dihentikan...")
        sys.exit()
