import gym_tetris

import gym
import torch.nn as nn
import matplotlib.pyplot as plt

from torch import FloatTensor, float32
from gym_tetris.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import (
    ResizeObservation,
    GrayScaleObservation,
    NormalizeObservation,
    FrameStack,
    
)
from torchvision.transforms import v2

from src.utils import make_env, preprocessing

# 4 hidden convolutional layers, where each convolves 32 filters with a kernel of size 3 by 3, with stride set to the value of 2 and padding set to one.
class SimpleConvNet(nn.Module):
    def __init__(self, input_channels=4):
        super(SimpleConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels, out_channels=32, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
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
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook_fn))

    # Melewatkan input melalui model
    _ = model(input_tensor)

    # Menghapus hooks
    for hook in hooks:
        hook.remove()

    # Visualisasi output
    fig, axes = plt.subplots(
        1, len(layer_outputs) + 1, figsize=(18, 12)
    )  # Membuat 2x3 subplot
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


env = gym_tetris.make("TetrisA-v3", apply_api_compatibility=True, render_mode="human")
env.metadata["render_modes"] = ["rgb_array", "human"]
env.metadata["render_fps"] = 60
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = NormalizeObservation(env)
env = ResizeObservation(env, 84)
env = FrameStack(env, 4)
env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    if done:
        break

# state = FloatTensor(obs.__array__().squeeze(3)).unsqueeze(0)
# model = SimpleConvNet(input_channels=4)
# visualize_conv_output(model, state)

# def plot_observation(observation):
#     # observasi adalah array dengan shape (4, 84, 84)
#     num_frames = observation.shape[0]
#     fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
    
#     for i in range(num_frames):
#         axes[i].imshow(observation[i], cmap='gray')  # Gunakan cmap 'gray' untuk grayscale
#         axes[i].set_title(f'Frame {i + 1}')
#         axes[i].axis('off')  # Nonaktifkan sumbu
    
#     plt.tight_layout()
#     plt.show()

# # Plot gambar observasi
# plot_observation(FloatTensor(obs.__array__().squeeze(3)))