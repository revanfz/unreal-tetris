import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        input_layers = [
            nn.Conv2d(num_inputs, 16, 8, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ]
        self.convolutional = nn.Sequential(*input_layers)
        self.fc = nn.Sequential(
            nn.Linear(32 * 10 * 10, 256),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTMCell(256, 256)
        self.critic = nn.Linear(256, 1)
        self.actor = nn.Linear(256, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        x = self.convolutional(x)
        x = self.fc(x.view(-1, 32 * 10 * 10))
        hx, cx = self.lstm(x, (hx, cx))
        return self.actor(hx), self.critic(hx), hx, cx
