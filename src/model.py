import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from torch import zeros, Tensor, device


class SharedNetwork(nn.Module):
    def __init__(self, input_channels: int, hidden_size: int):
        super(SharedNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 6 * 6, hidden_size)

    def forward(self, x: Tensor, hidden: Optional[tuple[Tensor, Tensor]]):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, hidden)
        return hx, cx


class Actor(nn.Module):
    def __init__(self, hidden_size: int, num_actions: int):
        super(Actor, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_actions)

    def forward(self, x: Tensor):
        x = F.relu(self.fc(x))
        policy = self.output(x)
        return F.softmax(policy, dim=-1)


class Critic(nn.Module):
    def __init__(self, hidden_size: int):
        super(Critic, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor):
        x = F.relu(self.fc(x))
        return self.output(x)


class ActorCriticNetwork(nn.Module):
    def __init__(
        self, input_channels: int, num_actions: int, hidden_size: int, device: device
    ):
        super(ActorCriticNetwork, self).__init__()
        self.device = device
        self.shared = SharedNetwork(
            input_channels=input_channels, hidden_size=hidden_size
        )
        self.actor = Actor(num_actions=num_actions, hidden_size=hidden_size)
        self.critic = Critic(hidden_size=hidden_size)
        self.to(device)

    def forward(self, x: Tensor, hidden: tuple[Tensor, Tensor]):
        x = x.to(self.device)
        hx, cx = self.shared(x, hidden)
        actor_output = self.actor(hx)
        critic_output = self.critic(hx)
        return actor_output, critic_output, (hx, cx)
