import torch
import torch.nn.functional as F

from pprint import pp
from model import UNREAL
from utils import make_env, preprocessing


params = dict(
    alpha=0.00029,
    gamma=0.94257,
    beta=0.00067,
    tau=0.09855
)

if __name__ == "__main__":
    env = make_env(
        grayscale=False, framestack=None, resize=None, render_mode="human"
    )
    device = torch.device("cpu")

    model =  UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=env.action_space.n,
        hidden_size=256,
        device=device,
        beta=params["beta"],
        gamma=params["gamma"]
    )
    model.load_state_dict(
        torch.load(
            "trained_models/a3c_tetris.pt",
            weights_only=True,
        ),
    )
    model.eval()

    done = True

    # for i in range(num_tests):
    while True:
        if done:
            state, info = env.reset()
            state = preprocessing(state)
            prev_action = torch.zeros(1, env.action_space.n).to(device)
            prev_reward = torch.zeros(1, 1).to(device)
            hx = torch.zeros(1, 256).to(device)
            cx = torch.zeros(1, 256).to(device)
            episode_reward = 0
            prev_lines = 0
        else:
            hx = hx.data
            cx = cx.data
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            policy, _, _, hx, cx = model(
                state_tensor, prev_action, prev_reward, (hx, cx)
            )

            action = policy.cpu().argmax().unsqueeze(0)

            next_state, reward, done, _, info = env.step(action.item())
            if info["number_of_lines"] > prev_lines:
                reward += 10 * (info["number_of_lines"] - prev_lines)
                prev_lines = info["number_of_lines"]
            episode_reward += reward
            next_state = preprocessing(next_state)
            prev_action = F.one_hot(action, num_classes=env.action_space.n).to(
                device
            )
            prev_reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
            state = next_state
        
        if done:
            pp(episode_reward)