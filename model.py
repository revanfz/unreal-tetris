import torch
import numpy as np
import torch.nn as nn

from torch.cuda import is_available
from torch import Tensor, zeros


class ActorCriticFF(nn.Module):
    def __init__(self, num_inputs: tuple, num_actions: int) -> None:
        super(ActorCriticFF, self).__init__()
        self.device = "cuda" if is_available() else "cpu"
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
        self.fc = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU())
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)
        self._initialize_weights()
        
    def _get_conv_out_size(self, shape):
        o = self.conv_net(torch.zeros((1, *shape)))
        return int(np.prod(o.size()))
    
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv_net(x)
        x = x.view(-1, int(np.prod(x.size())))
        x = self.fc(x)
        return (self.actor(x), self.critic(x))
    
    def calculate_loss(
        self,
        done: bool,
        values: Tensor,
        log_probs: Tensor,
        entropies: Tensor,
        rewards: Tensor,
        gamma: float,
        beta: float,
    ) -> Tensor:
        """
        Menghitung loss
        """
        T = len(rewards)
        td_errors = zeros(T, device=self.device)
        R = values[-1] * int(not done)

        for t in reversed(range(T)):
            R = rewards[t] + gamma * R
            td_errors[t] = R

        advantage = td_errors - values
        critic_loss = advantage.pow(2).mean()
        actor_loss = -(log_probs * advantage).mean() - beta * entropies.mean()
        total_loss = actor_loss + 0.5 * critic_loss

        return total_loss


class ActorCriticLSTM(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCriticLSTM, self).__init__()
        self.device = "cuda" if is_available() else "cpu"
        self.conv_net = nn.Sequential(
            nn.Conv2d(num_inputs[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out_size((num_inputs))
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        self.lstm = nn.LSTMCell(512, 256)
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

        self._initialize_weights()

    def _get_conv_out_size(self, shape):
        o = self.conv_net(torch.zeros((1, *shape)))
        return int(np.prod(o.size()))

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        x = self.conv_net(x)
        x = x.view(-1, int(np.prod(x.size())))
        x = self.fc(x)
        hx, cx = self.lstm(x, (hx, cx))
        return (self.actor(hx), self.critic(hx), hx, cx)