import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        # self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(num_inputs, 16, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 10 * 10, 512)
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)
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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # hx, cx = self.lstm(x.view(-1, 32 * 6 * 6), (hx, cx))
        hx, cx = self.lstm(x.view(-1, 32 * 10 * 10), (hx, cx))
        return self.actor(hx), self.critic(hx), hx, cx
