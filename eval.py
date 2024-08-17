import os
os.environ["OMP_NUM_THREADS"] = "1"

import gym.wrappers.record_video

import gym
import time
import torch
import logging
import argparse
import gym_tetris
import torch.nn.functional as F

from model import ActorCriticLSTM
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack, GrayScaleObservation, RecordEpisodeStatistics
from utils import setup_logger, preprocessing


gym.logger.set_level(40)

parser = argparse.ArgumentParser(description="A3C_EVAL")
parser.add_argument(
    "-ne",
    "--num-episodes",
    type=int,
    default=100,
    help="how many episodes in evaluation (default: 100)",
)
parser.add_argument(
    "-lmd",
    "--load-model-dir",
    default="trained_models/",
    help="folder to load trained models from",
)
parser.add_argument(
    "-ld",
    "--log-dir",
    default="logs/",
    help="folder to log model evaluation",
)
parser.add_argument("-l", "--level", default="0", help="Configure speed level (cap 29)")
parser.add_argument("-gpu", "--use-gpu", default=False, help="Run on GPU/CPU")
args = parser.parse_args()

saved_state = torch.load(
    f"{args.load_model_dir}baseline_a3c_tetris.pt",
    map_location=lambda storage, loc: storage,
    # f"{args.load_model_dir}tuned_a3c_tetris.pt", map_location=lambda storage, loc: storage
)

setup_logger(f"A3C_Tetris_eval_log", rf"{args.log_dir}A3C_Tetris_eval_log")
log = logging.getLogger(f"A3C_Tetris_eval_log")

d_args = vars(args)
for k in d_args.keys():
    log.info(f"{k}: {d_args[k]}")

env = gym_tetris.make("TetrisA-v3", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, MOVEMENT)
env = GrayScaleObservation(env)
env = FrameStack(env, 4)
env = RecordEpisodeStatistics(env, deque_size=10)

device = "cuda" if args.use_gpu else "cpu"
num_tests = 0
start_time = time.time()
reward_total_sum = 0

model = ActorCriticLSTM((4, 84, 84), env.action_space.n).to(device)
model.load_state_dict(saved_state)
model.eval()

try:
    num_tests = 0
    sum_rewards = 0    
    done = True
    for i_episode in range(args.num_episodes):
        while True:
            with torch.no_grad():
                if done:
                    state, info = env.reset()
                    hx = torch.zeros(1, 256).to(device)
                    cx = torch.zeros(1, 256).to(device)
                else:
                    hx = hx.data
                    cx = cx.data

                state = preprocessing(state.copy()).to(device)
                policy, _, hx, cx = model(state.unsqueeze(0), hx, cx)
                probs = F.softmax(policy, dim=1)
                action = probs.cpu().numpy().argmax()
            state, reward, done, _, info = env.step(action.item())

            if done:
                num_tests += 1
                sum_rewards += info['episode']['r']
                reward_mean = sum_rewards / num_tests
                log.info(
                    f'Time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))}, episode reward {info["episode"]["r"]}, episode length {info["episode"]["l"]}, reward mean {reward_mean:.4f}'
                )
                break

except KeyboardInterrupt:
    print("KeyboardInterrupt exception is caught")

finally:
    print("gym evalualtion process finished")

env.close()
