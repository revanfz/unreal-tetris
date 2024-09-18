
import cv2
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch import Tensor, float32, FloatTensor
from torchvision.transforms import v2
from torchvision.utils import save_image

from utils import make_env

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
            nn.Conv2d(input_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
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
        if isinstance(layer, nn.Conv2d):
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



def plot_observation(observation):
    # observasi adalah array dengan shape (4, 84, 84)
    plt.imshow(observation)
    plt.axis('off')  # Mematikan sumbu untuk tampilan gambar yang lebih bersih
    
    plt.tight_layout()
    plt.show()

# Create model and sample input

# Contoh penggunaan

env = make_env(grayscale=False, resize=None, normalize=True, framestack=None)
env.reset()
model = SimpleConvNet(input_channels=env.observation_space.shape[-1])

for i in range(4):
    action = env.action_space.sample()
    state, reward, done, _, info = env.step(action)
    if done:
        break

# img_path = "./input.png"
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_arr = np.array(img)
# state = torch.FloatTensor(preprocess_frame_stack(img_arr)).unsqueeze(0)
# state = pixel_control(state)
state = FloatTensor(state)
print(state.shape)
state = preprocessing(FloatTensor(state).permute(2, 0, 1)).unsqueeze(0)
print(state.shape)

# Plot gambar observasi
visualize_conv_output(model, state)
