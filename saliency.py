import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from model import UNREAL
from utils import make_env, preprocessing

def compute_saliency_map(model, state, action, reward, hidden_state):
    state = state.requires_grad_()
    policy, _, hx, cx = model(state, action, reward, hidden_state)
    chosen_action = policy.argmax(dim=1)
    
    # Backpropagate hanya untuk action yang dipilih
    model.zero_grad()
    policy[0, chosen_action].backward()

    # Ambil absolut gradien dari input
    saliency, _ = torch.max(state.grad.abs(), dim=1)
    return saliency.squeeze().cpu().numpy()


device = torch.device("cpu")
checkpoint = torch.load("trained_models/UNREAL-cont-fine-tuning.pt", weights_only=True, map_location=torch.device("cpu"))
saliency = []
frameskip = [5, 4, 3, 2]

model = UNREAL(
    n_inputs=(84, 84, 3),
    n_actions=6,
    hidden_size=256,
    device=device,
)
model.load_state_dict(checkpoint)
model.eval()
for level in range(10, 20, 1):
    env = make_env(
        record=True,
        resize=84,
        level=level,
        id="TetrisA-v3",
        render_mode="rgb_array",
        skip=frameskip[(level-10)//3]
    )
    obs, info = env.reset(seed=17)
    state = preprocessing(obs)
    action = torch.nn.functional.one_hot(
        torch.LongTensor([0]), num_classes=env.action_space.n
    )
    reward = torch.zeros(1, 1)
    hx = torch.zeros(1, 256)
    cx = torch.zeros(1, 256)

    while sum(info["statistics"].values()) < 20:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        policy, _, hx, cx = model(state_tensor, action, reward, (hx, cx))
        action = policy.argmax().unsqueeze(0)
        next_state, reward, done, _, info = env.step(action.item())
        next_state = preprocessing(next_state)
        action = torch.nn.functional.one_hot(action, num_classes=env.action_space.n)
        reward = torch.FloatTensor([reward]).unsqueeze(0)
        state = next_state

    # Masukin model RL lo
    # image = cv2.imread("level_11.png")  # Masukin gambar dari level 11
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert ke grayscale kalau perlu
    # image = transforms.ToTensor()(image).unsqueeze(0)

    # Hitung saliency map
    saliency.append(compute_saliency_map(model, state_tensor, action, reward, (hx, cx)))
    env.close()

# Tampilkan hasil
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 12))
axes = axes.flatten()  # Mengubah axes menjadi array 1D untuk memudahkan iterasi
for level in range(10, 20, 1):
    axes[level-10].imshow(saliency[level-10], cmap='hot')
    axes[level-10].set_title(f'Saliency level {level}')
    axes[level-10].set_xticks([])
    axes[level-10].set_yticks([])
plt.tight_layout()
plt.axis('off')
plt.show()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import cosine

# def get_histogram(image_path):
#     img = cv2.imread(image_path)  # Baca gambar
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert ke RGB
#     hist = cv2.calcHist([img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])  # Histogram 3D
#     return hist.flatten() / hist.sum()  # Normalisasi

# # Masukin path gambar setiap level
# level_images = {
#     "11": "./img/appendix/Level 11-26.png",
#     "12": "./img/appendix/Level 12-26.png",
#     "15": "./img/appendix/Level 15-26.png",
#     "17": "./img/appendix/Level 17-26.png",
#     "18": "./img/appendix/Level 18-26.png",
#     "19": "./img/appendix/Level 19-26.png"
# }

# # Hitung histogram tiap level
# histograms = {level: get_histogram(img_path) for level, img_path in level_images.items()}

# # Bandingin level 11 vs level latih (12, 15, 18, 19)
# for trained_level in ["12", "15", "18", "19"]:
#     similarity = 1 - cosine(histograms["11"], histograms[trained_level])
#     print(f"Similarity warna Level 11 vs {trained_level}: {similarity:.4f}")

# # Bandingin level 17 vs level latih (12, 15, 18, 19)
# for trained_level in ["12", "15", "18", "19"]:
#     similarity = 1 - cosine(histograms["17"], histograms[trained_level])
#     print(f"Similarity warna Level 17 vs {trained_level}: {similarity:.4f}")