import gym_tetris

import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from model import UNREAL
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import MOVEMENT
from torch import Tensor, float32, stack
from torchvision.transforms import v2
from torchvision.utils import save_image

def preprocessing(state: np.ndarray) -> Tensor:
    preprocess = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(float32, scale=True),
            v2.Lambda(lambda x: v2.functional.crop(x, 48, 96, 160, 80)),
            v2.Resize((84, 84))
        ]
    )
    return preprocess(state)

def pixel_control(state: Tensor) -> Tensor:
    return v2.CenterCrop(80)(state)


def preprocess_frame_stack(frame_stack: np.ndarray, save: bool = False) -> Tensor:
    # frame_stack = np.transpose(frame_stack, (2, 0, 1))
    if save:
        cv2.imwrite("input.png", cv2.cvtColor(frame_stack, cv2.COLOR_BGR2RGB))

    return preprocessing(frame_stack.copy()).numpy()


class SimpleConvNet(nn.Module):
    def __init__(self, input_channels=3):
        super(SimpleConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            # nn.Conv2d(input_channels, 16, kernel_size=8, stride=4),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(16, 32, kernel_size=4, stride=2),
            # nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=4, stride=4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_layers(x)

def visualize_conv_output(model, input_tensor):
    # Daftar untuk menyimpan output dari setiap lapisan
    layer_outputs = []

    # Hook function untuk mengambil output dari setiap lapisan
    def hook_fn(module, input, output):
        layer_outputs.append(output.detach())

    # Mendaftarkan hook untuk setiap lapisan konvolusi
    hooks = []
    for layer in model.conv_layers:
        if isinstance(layer, nn.AvgPool2d):
            hooks.append(layer.register_forward_hook(hook_fn))

    # Melewatkan input melalui model
    _ = model(input_tensor)

    # Menghapus hooks
    for hook in hooks:
        hook.remove()

    # Visualisasi output
    fig, axes = plt.subplots(1, len(layer_outputs) + 1, figsize=(18, 12))  # Membuat 2x3 subplot
    axes = axes.flatten()

    # Plot gambar input
    input_img = (
        input_tensor[0].permute(1, 2, 0).cpu().numpy()
    )  # Mengubah dimensi ke HWC
    axes[0].imshow(input_img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Plot output dari setiap lapisan konvolusi
    for idx, output in enumerate(layer_outputs):
        print(f"Shape of output: {output.shape}")

        img = output[0, 0].cpu().numpy()  # Mengambil channel pertama dari output
        axes[idx + 1].imshow(img, cmap="viridis")
        axes[idx + 1].set_title(f"Conv Layer {idx+1} Output")
        axes[idx + 1].axis("off")

    plt.tight_layout()
    plt.show()


# Create model and sample input

# Contoh penggunaan

env = gym_tetris.make(
    "TetrisA-v4",
    apply_api_compatibility=True,
    render_mode="rgb_array",
)
env = JoypadSpace(env, MOVEMENT)
env.metadata["render_modes"] = ["rgb_array", "human"]
env.metadata["render_fps"] = 60
_, _ = env.reset()

model = SimpleConvNet(input_channels=3)

for i in range(2000):
    action = env.action_space.sample()
    state, reward, done, _, info = env.step(0)
    state, reward, done, _, info = env.step(env.action_space.sample())
img_path = "./input.png"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_arr = np.array(img)
state = torch.FloatTensor(preprocess_frame_stack(img_arr)).unsqueeze(0)
state = pixel_control(state)
print(state.shape)
visualize_conv_output(model, state)
