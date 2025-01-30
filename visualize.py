
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision.transforms import v2
from torch import Tensor, float32, FloatTensor

from utils import make_env

def preprocessing(state: np.ndarray) -> Tensor:
    preprocess = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(float32, scale=True),
            # v2.Lambda(lambda x: v2.functional.crop(x, 48, 96, 160, 80)),
            # v2.Resize((84, 84))
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

class SimpleDeconv(nn.Module):
    def __init__(self, hidden_size, n_actions):
        super(SimpleDeconv, self).__init__()        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
        )
        self.fc_layer1 = nn.Sequential(
            nn.Linear(2048, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.lstm_layer = nn.LSTMCell(hidden_size + n_actions + 1, hidden_size)
        # Fully connected layer
        self.fc_layer2 = nn.Sequential(
            nn.Linear(hidden_size, 32 * 7 * 7), 
            nn.ReLU(inplace=True)
        )
        
        # Dekonvolusi spasial
        self.deconv_spatial = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3), 
            nn.ReLU(inplace=True)
        )
        
        # Dekonvolusi untuk nilai
        self.deconv_value = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2)
        
        # Dekonvolusi untuk keuntungan
        self.deconv_advantage = nn.ConvTranspose2d(32, n_actions, kernel_size=4, stride=2)
    
    def forward(self, observation: torch.Tensor, reward: torch.Tensor, action: torch.Tensor, hidden_state: tuple[torch.Tensor, torch.Tensor]):
        # Lakukan feedforward melalui fc_layer
        x = self.conv_layer(observation)
        x = self.fc_layer1(x.reshape(x.size(0), -1))
        x = torch.cat([x, action, reward], dim=1)
        hx, cx = self.lstm_layer(x, hidden_state)


        x = self.fc_layer2(hx)  # Bentuk tensor menjadi [batch_size, 32*7*7]
        x = x.view(-1, 32, 7, 7)  # Ubah bentuk menjadi [batch_size, 32, 7, 7]
        
        # Dekonvolusi spasial
        x_spatial = self.deconv_spatial(x)
        
        # Dekonvolusi untuk nilai dan keuntungan
        value = self.deconv_value(x_spatial)
        advantage = self.deconv_advantage(x_spatial)
        
        return value, advantage, x_spatial
    

def plot_images(value, advantage, spatial):
    # Print shapes untuk debugging
    print(f"Value shape: {value.shape}")
    print(f"Advantage shape: {advantage.shape}")
    print(f"Spatial shape: {spatial.shape}")
    
    # Ambil batch pertama jika ada batch dimension
    if len(value.shape) == 4:
        value = value[0]
    if len(advantage.shape) == 4:
        advantage = advantage[0]
    if len(spatial.shape) == 4:
        spatial = spatial[0]
        
    # Hitung Q-values
    advantage_mean = advantage.mean(dim=0, keepdim=True)  # Mean across actions
    q_aux = value + (advantage - advantage_mean)  # [n_actions, H, W]
    q_max, _ = torch.max(q_aux, dim=0)  # [H, W]
    
    # Konversi ke numpy
    q_max_np = q_max.cpu().detach().numpy()
    q_aux_np = q_aux.cpu().detach().numpy()
    
    # Plot gambar pertama (Q-max)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(q_max_np, cmap='viridis')
    plt.colorbar(im)
    plt.title('Max Q-values per pixel')
    plt.tight_layout()
    plt.show()
    
    # Plot Q-values untuk setiap action
    n_actions = advantage.shape[0]
    rows = (n_actions + 1) // 2
    fig, axes = plt.subplots(2, rows, figsize=(20, 10))
    axes = axes.ravel()
    
    for i in range(n_actions):
        im = axes[i].imshow(q_aux_np[i], cmap='viridis')
        axes[i].set_title(f'Q-values Action {i}')
        plt.colorbar(im, ax=axes[i])
        
    # Sembunyikan subplot yang tidak terpakai
    for i in range(n_actions, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

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
        input_tensor[0].permute(1, 2, 0).cpu().detach().numpy()
    )  # Mengubah dimensi ke HWC
    axes[0].imshow(input_img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Plot output dari setiap lapisan konvolusi
    for idx, output in enumerate(layer_outputs):
        print(f"Shape of output: {output.shape}")

        img = output[0, 0].cpu().detach().numpy()  # Mengambil channel pertama dari output
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

env = make_env(grayscale=False, resize=84, framestack=None, skip=2)
env.reset()
model = SimpleConvNet(input_channels=env.observation_space.shape[-1])

for i in range(4):
    action = env.action_space.sample()
    state, reward, done, _, info = env.step(action)

    # Plot menggunakan matplotlib
    # plt.imshow(state, cmap='gray')  # Gunakan 'gray' untuk gambar grayscale
    # plt.axis('off')  # Untuk menghilangkan sumbu (opsional)
    # plt.show()
    # state = preprocessing(state).unsqueeze(0)

    # Plot gambar observasi
    # visualize_conv_output(model, state)

    if done:
        break

hidden_size = 256
n_actions = 12
model = SimpleDeconv(hidden_size, n_actions)

# Membuat input dummy dengan hidden_size
state_tensor = torch.from_numpy(state.transpose(2, 0, 1)[:, 2:-2, 2:-2]).float()
action_oh = torch.nn.functional.one_hot(torch.tensor(action).long(), num_classes=12).unsqueeze(0)
reward = torch.tensor([[reward]]).float()
value, advantage, spatial = model(state_tensor.unsqueeze(0), reward, action_oh, None)

# Visualisasikan output dari dekonvolusi
plot_images(value, advantage, spatial)

# img_path = "./input.png"
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_arr = np.array(img)
# state = torch.FloatTensor(preprocess_frame_stack(img_arr)).unsqueeze(0)
# state = pixel_control(state)

# # Plot menggunakan matplotlib
# plt.imshow(state, cmap='gray')  # Gunakan 'gray' untuk gambar grayscale
# plt.axis('off')  # Untuk menghilangkan sumbu (opsional)
# plt.show()
# state = preprocessing(state.copy()).unsqueeze(0)

# # Plot gambar observasi
# visualize_conv_output(model, state)
