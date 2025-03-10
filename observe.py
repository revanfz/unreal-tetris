import time
import torch
import torch.nn.functional as F

from model import UNREAL
from torchvision import transforms
from torch.distributions import Categorical
from utils import make_env, preprocessing, pixel_diff

# params = dict(
#     lr=0.0005,
#     unroll_steps=20,
#     beta=0.00067,
#     gamma=0.99,
#     hidden_size=256,
#     task_weight=0.01,
# )
params = dict(
    lr=0.0003,
    unroll_steps=20,
    beta=0.01,
    gamma=0.99,
    hidden_size=256,
    task_weight=1.0,
)

device = torch.device("cpu")

if __name__ == "__main__":
    env = make_env(id="TetrisA-v3", resize=84, render_mode="human", level=10, skip=2)
    # checkpoint = torch.load("trained_models/UNREAL-cont.pt", weights_only=True)
    checkpoint = torch.load("trained_models/UNREAL-cont-fine-tuning_checkpoint.tar", weights_only=True)
    local_model = UNREAL(
        n_inputs=(84, 84, 3),
        n_actions=env.action_space.n,
        hidden_size=256,
        device=device,
    )
    local_model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    local_model.eval()

    done = True
    action = F.one_hot(torch.LongTensor([0]), num_classes=env.action_space.n).to(device)
    reward = torch.zeros(1, 1).to(device)


    while True:
        if done:
            blocks = 0
            state, info = env.reset()
            state = preprocessing(state)
            hx = torch.zeros(1, params["hidden_size"]).to(device)
            cx = torch.zeros(1, params["hidden_size"]).to(device)
            eps_r = []
            step = 0
            ep_r = 0
        else:
            hx = hx.detach()
            cx = cx.detach()
        
        
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        policy, value, hx, cx = local_model(state_tensor, action, reward, (hx, cx))

        dist = Categorical(probs=policy)
        # action = dist.sample()
        # if policy[0][1] >= 0.1 or policy[0][2] >= 0.1:
        # print("probs:", policy)
        action = policy.argmax().unsqueeze(0)
        # action = torch.tensor(env.action_space.sample()).unsqueeze(0)
    
        next_state, reward, done, _, info = env.step(action.item())
        if reward != 0:
            print(reward)

        # if sum(info['statistics'].values()) > 1:
        #     print(env.unwrapped._board)
        #     break
        curr_block = sum(info["statistics"].values())
            
        next_state = preprocessing(next_state)
        pixel_change = pixel_diff(state, next_state)
        action = F.one_hot(action, num_classes=env.action_space.n).to(device)
        reward = torch.FloatTensor([[reward]]).to(device)
        state = next_state
        step += 1

        if done:
            print(info)