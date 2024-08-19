import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


class ActorCriticFF(nn.Module):
    def __init__(self, num_inputs: tuple, num_actions: int) -> None:
        super(ActorCriticFF, self).__init__()
        input_layers = [
            nn.Conv2d(num_inputs[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        ]
        self.conv_net = nn.Sequential(*input_layers)
        conv_out_size = self._get_conv_out_size((num_inputs))
        self.fc = nn.Sequential(nn.Linear(conv_out_size, 256), nn.ReLU())
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

    def _get_conv_out_size(self, shape: tuple) -> int:
        o = self.conv_net(torch.zeros((1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_net(input)
        x = x.view(-1, int(np.prod(x.size())))
        x = self.fc(x)
        return (self.actor(x), self.critic(x))


class ActorCriticLSTM(nn.Module):
    def __init__(self, num_inputs: tuple, num_actions: int, hidden_size: int) -> None:
        super(ActorCriticLSTM, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs[0], 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        conv_out_size = self._get_conv_out_size((num_inputs))
        self.lstm = nn.LSTMCell(conv_out_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, input: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        normalized_hx = self.layer_norm(hx)

        return self.actor(normalized_hx), self.critic(normalized_hx), hx, cx

    def _get_conv_out_size(self, shape: tuple) -> int:
        o = self.conv1(torch.zeros((1, *shape)))
        o = self.conv2(o)
        return int(np.prod(o.size()))
