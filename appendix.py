import torch
import torch.nn.functional as F

from model import UNREAL
from PIL import Image as im
from utils import make_env, preprocessing

if __name__ == "__main__":
    checkpoint = torch.load("trained_models/UNREAL-cont-fine-tuning.pt", weights_only=True, map_location=torch.device("cpu"))

    model = UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=6,
        hidden_size=256,
        device=torch.device("cpu"),
    )
    model.load_state_dict(checkpoint)
    model.eval()

    for level in [11, 17]:
        env = make_env(resize=84, level=level, skip=2)
        env.reset()

        terminated = False
        state, info = env.reset()
        state = preprocessing(state)
        action = F.one_hot(
            torch.LongTensor([0]), num_classes=env.action_space.n
        )
        reward = torch.zeros(1, 1)
        hx = torch.zeros(1, 256)
        cx = torch.zeros(1, 256)
        blocks = 0

        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy, _, hx, cx = model(state_tensor, action, reward, (hx, cx))
            action = policy.argmax().unsqueeze(0)

            obs, reward, terminated, truncated, info = env.step(action.item())
            block_placed = sum(info["statistics"].values())
            if block_placed >= 25 and block_placed > blocks:
                # frame = im.fromarray(env.render())
                frame = im.fromarray(obs)
                frame.save(f"./img/appendix/Level {level}-{blocks}.png")
            blocks = block_placed

            if terminated:
                break

            next_state = preprocessing(obs)
            action = F.one_hot(action, num_classes=env.action_space.n)
            reward = torch.FloatTensor([reward]).unsqueeze(0)
            state = next_state
