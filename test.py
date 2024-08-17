from multiprocessing.synchronize import Event
import time
import torch
import logging
import gym_tetris
import torch.nn.functional as F

from model import ActorCriticFF, ActorCriticLSTM
from optimizer import SharedAdam
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from multiprocessing.sharedctypes import Synchronized
from gym.wrappers import FrameStack, GrayScaleObservation, RecordEpisodeStatistics
from utils import preprocess_frame_stack, preprocessing, setup_logger


def save_model(
    steps: int, path: str, global_model: ActorCriticLSTM, optimizer: SharedAdam, num_tests: int
):
    torch.save(
        {
            "steps": steps,
            "num_tests": num_tests,
            "model_state_dict": global_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "{}/baseline_a3c_tetris.tar".format(path),
        # "{}/tuned_a3c_tetris.tar".format(path)
    )


def test_model(
    opt: dict,
    global_model: ActorCriticLSTM,
    optimizer: SharedAdam,
    global_steps: Synchronized,
    num_tests: int
):
    try:
        device = "cuda" if opt.use_cuda else "cpu"
        writer = SummaryWriter(opt.log_test_path)
        setup_logger("A3C_Tetris_log", rf"logs/A3C_Tetris_baseline_log")
        log = logging.getLogger(f"A3C_Tetris_log")

        d_args = vars(opt)
        for k in d_args.keys():
            log.info(f"{k}: {d_args[k]}")

        torch.manual_seed(42)
        env = gym_tetris.make(
            "TetrisA-v3", apply_api_compatibility=True, render_mode="human"
        )
        env = JoypadSpace(env, MOVEMENT)
        env = GrayScaleObservation(env)
        env = FrameStack(env, 4)
        env.metadata['render_modes'] = ["rgb_array", "human"]
        env.metadata['render_fps'] = 60

        local_model = ActorCriticLSTM((4, 84, 84), env.action_space.n)
        local_model.eval()

        dummy_input = (
            torch.zeros(1, 4, 84, 84),
            torch.zeros(1, 256),
            torch.zeros(1, 256),
        )
        writer.add_graph(local_model, dummy_input, False)
        writer.close()
        local_model.to(device)

        start_time = time.time()
        episode_reward = 0
        sum_rewards = 0
        curr_step = 0
        max_score = 0
        done = True

        while global_steps.value <= opt.max_steps:
            if done:
                local_model.load_state_dict(global_model.state_dict())

            with torch.no_grad():
                if done:
                    state, info = env.reset()
                    hx = torch.zeros(1, 256).to(device)
                    cx = torch.zeros(1, 256).to(device)
                else:
                    hx = hx.data
                    cx = cx.data

                state = preprocess_frame_stack(state).to(device)
                policy, _, hx, cx = local_model(state.unsqueeze(0), hx, cx)
                probs = F.softmax(policy, dim=1)
                action = probs.cpu().numpy().argmax()
            state, reward, done, _, _ = env.step(action.item())
            episode_reward += reward
            # env.render()
            curr_step += 1

            if global_steps.value % opt.save_interval == 0:
                save_model(global_steps.value, opt.model_path, global_model, optimizer, num_tests)

            if done:
                num_tests += 1
                sum_rewards += episode_reward
                reward_mean = sum_rewards / num_tests
                log.info(
                    f'Time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))}, episode reward {episode_reward}, episode length {curr_step}, reward mean {reward_mean:.4f}'
                )
                writer.add_scalar(f"Global/Mean rewards", reward_mean, num_tests)
                for name, weight in local_model.named_parameters():
                    writer.add_histogram(name, weight, num_tests)

                if episode_reward >= max_score:
                    max_score = episode_reward
                    save_model(
                        global_steps.value, opt.model_path, global_model, optimizer, num_tests
                    )

                episode_reward = 0
                curr_step = 0

    except KeyboardInterrupt:
        time.sleep(0.01)
        print("KeyboardInterrupt exception is caught")

    finally:
        env.close()
        writer.close()
