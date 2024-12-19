import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch.multiprocessing as mp

from model import UNREAL
from utils import make_env, update_progress
from torch.nn.functional import one_hot
from torch.distributions import Categorical
from gymnasium.wrappers import RecordEpisodeStatistics
from multiprocessing.sharedctypes import SynchronizedBase
from torch import device, load, from_numpy, tensor, zeros, zeros_like, no_grad


TEST_CASE = 10
EVAL_PATH = "./"
TEST_LENGTH = 1_000_000
MODEL_PATH = "./trained_models/UNREAL.pt"
CSV_PATH = "./UNREAL-eval"
if not os.path.isdir(CSV_PATH):
    os.makedirs(CSV_PATH)
CHECKPOINT = load(MODEL_PATH, weights_only=True)
DEVICE = device("cpu")
MODEL = UNREAL(
    n_inputs=(84, 84, 3),
    n_actions=12,
    device=DEVICE,
)
MODEL.load_state_dict(CHECKPOINT)
MODEL.eval()


def agent(
    rank: int,
    test_case: int,
    max_steps: int,
    queue: mp.Queue,
    global_steps: SynchronizedBase,
    device: device,
):
    env = make_env(
        id="TetrisA-v0",
        resize=84,
        level=test_case,
        skip=4,
        render_mode="human" if not rank else "rgb_array",
    )
    env = RecordEpisodeStatistics(env, buffer_length=2000)

    done = True
    action = one_hot(tensor([0]).long(), num_classes=MODEL.n_actions).to(device)
    reward = zeros(1, 1, device=device).float()

    with no_grad():
        while global_steps.value <= max_steps:
            if done:
                action_taken = list()
                state, info = env.reset()
                hx = zeros(1, 256, device=device)
                cx = zeros_like(hx, device=device)

            state = from_numpy(state.transpose(2, 0, 1) / 255.0).float().to(device)
            policy, _, hx, cx = MODEL(state.unsqueeze(0), action, reward, (hx, cx))

            # action = policy.argmax().unsqueeze(0)
            dist = Categorical(probs=policy)
            action = dist.sample()

            state, reward, done, _, info = env.step(action.item())
            action_taken.append(action.item())
            action = one_hot(action, num_classes=MODEL.n_actions).to(device)
            reward = tensor([[reward]], device=device).float()

            with global_steps.get_lock():
                global_steps.value += 1

            if done:
                queue.put(np.array([info['episode']['r'], sum(info["statistics"].values()), info['episode']['l'], info['episode']['t'], action_taken], dtype=object))
    queue.put(None)


if __name__ == "__main__":
    try:
        for test in range(TEST_CASE):
            shared_data = mp.Queue()
            global_steps = mp.Value("i", 0)

            processes = []
            all_data = []
            finished_agent = 0


            progress_process = mp.Process(
                target=update_progress,
                args=(
                    global_steps, TEST_LENGTH
                ),
                kwargs=(
                    {
                        "desc": f"Evaluating model case {test+1}",
                        "unit": "steps"
                    }
                ),
            )
            progress_process.start()
            processes.append(progress_process) 

            for rank in range(mp.cpu_count()):
                process = mp.Process(
                    target=agent,
                    args=(rank, test+10),
                    kwargs={
                        "device": DEVICE,
                        "max_steps": TEST_LENGTH,
                        "queue": shared_data,
                        "global_steps": global_steps,
                    },
                )
                process.start()
                processes.append(process)

            while finished_agent < mp.cpu_count():
                data = shared_data.get()
                if data is None:
                    finished_agent += 1
                    continue
                all_data.append(data)

            for process in processes:
                process.join()          

            data = np.array(all_data)
            df = pd.DataFrame(data, columns=["rewards", "blocks", "episode length", "survival time", "action sequence"])
            df.to_csv(f"{CSV_PATH}/{test}.csv", index=False)

    except KeyboardInterrupt as e:
        print(e)
        import sys
        sys.exit() 
