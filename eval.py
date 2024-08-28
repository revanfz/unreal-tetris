import gym_tetris
import torch
import torch.nn.functional as F

from model import UNREAL
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace

from utils import preprocess_frame_stack
from wrapper import ActionRepeatWrapper


if __name__ == "__main__":
    env = gym_tetris.make(
        "TetrisA-v3", render_mode="human", apply_api_compatibility=True
    )
    env = JoypadSpace(env, MOVEMENT)
    # env = ActionRepeatWrapper(env)
    device = torch.device("cpu")

    model = UNREAL((3, 84, 84), env.action_space.n, device=device)
    model.load_state_dict(
        torch.load(
            "trained_models/a3c_tetris.pt",
            weights_only=True,
        ),
    )
    model.eval()

    num_tests = 10
    total_rewards = 0
    done = False

    for i in range(num_tests):
        state, info = env.reset()
        state = preprocess_frame_stack(state)
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        prev_action = torch.zeros(1, env.action_space.n).to(device)
        prev_reward = torch.zeros(1, 1).to(device)
        episode_rewards = 0
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                policy, _, _, _, (hx, cx) = model(
                    state_tensor, prev_action, prev_reward, (hx, cx)
                )
                action: torch.Tensor = policy.argmax()

            # if action.item():
                # next_state, reward, done, _, info = env.step(0)
            next_state, reward, done, _, info = env.step(2)
            env.render()
            next_state = preprocess_frame_stack(next_state)

            prev_action = F.one_hot(action.unsqueeze(0), num_classes=env.action_space.n)
            prev_reward = torch.FloatTensor([reward]).unsqueeze(0)

            episode_rewards += reward
            state = next_state
        
        total_rewards += episode_rewards

    print(f"Mean reward: {total_rewards / num_tests}")
    