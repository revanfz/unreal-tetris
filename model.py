from typing import Union

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


class ConvNet(nn.Module):
    """
    Modul Konvolusi
    input:
        Gambar RGB 84x84 pixel
    process:
        1st filter : output 16 channel filter 8x8 stride 4
        2nd filter : output 32 channel filter 4x4 stride 2
        fc layer : output size = hidden size (default: 256)
    output:
        Tensor
    """

    def __init__(self, n_inputs: int, hidden_size: tuple):
        super(ConvNet, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(n_inputs[0], 16, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self._feature_size(n_inputs), hidden_size), nn.ReLU(inplace=True)
        )

    def _feature_size(self, n_inputs: int):
        with torch.no_grad():
            o = self.conv_layer(torch.zeros(1, *n_inputs))
        return int(np.prod(o.size()))

    def forward(self, observation: torch.Tensor):
        x = self.conv_layer(observation)
        x = x.view(x.size(0), -1)
        return self.fc_layer(x)


class LSTMNet(nn.Module):
    """
    Modul LSTM
    input:
        Tensor hasil konvolusi + n_actions + 1
        (gambar + jumlah aksi + reward)
    output:
        Tensor
    """

    def __init__(self, n_actions: int, hidden_size: int):
        super(LSTMNet, self).__init__()
        self.n_actions = n_actions
        self.lstm_layer = nn.LSTMCell(256 + n_actions + 1, hidden_size)

    def forward(self, conv_feature, hidden):
        lstm_feature, new_hidden = self.lstm_layer(conv_feature, hidden)
        return lstm_feature, (lstm_feature, new_hidden)


class ActorCritic(nn.Module):
    """
    Modul Actor Critic untuk menentukan Policy dan Value
    input:
        output lapisan LSTM
    output:
        Policy: peluang semua aksi (Tensor)
        Value: nilai estimasi (Tensor)
    """

    def __init__(self, n_actions: int, hidden_size: int):
        super(ActorCritic, self).__init__()
        self.n_actions = n_actions
        self.policy_layer = nn.Linear(hidden_size, n_actions)
        self.value_layer = nn.Linear(hidden_size, 1)

    def forward(self, lstm_feature: torch.Tensor):
        policy = F.softmax(self.policy_layer(lstm_feature), dim=-1)
        value = self.value_layer(lstm_feature)
        return policy, value


class PixelControl(nn.Module):
    """
    Modul pixel control
    Mengukur nilai pergantian pixel pada gambar hasil observasi
    input:
        Tensor hasil LSTM Gambar observasi yang di-crop jadi 80x80
    output:
        nilai Q_aux (Tensor)
    """

    def __init__(self, n_actions: int, hidden_size: int):
        super(PixelControl, self).__init__()
        self.n_actions = n_actions

        self.fc_layer = nn.Sequential(
            nn.Linear(hidden_size, 32 * 7 * 7), nn.ReLU(inplace=True)
        )
        self.deconv_value = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2)
        self.deconv_advantage = nn.ConvTranspose2d(
            32, n_actions, kernel_size=4, stride=2
        )

    def forward(self, lstm_feature: torch.Tensor):
        x = self.fc_layer(lstm_feature).view(-1, 32, 7, 7)
        value = self.deconv_value(x)
        advantage = self.deconv_advantage(x)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_aux = value + advantage - advantage_mean
        return q_aux


class FeatureControl(nn.Module):
    """
    Modul feature control
    Mengukur seberapa sering hidden unit diaktivasi selama pelatihan
    input:
        Tensor hasil LSTM Gambar observasi
    output:
        Q_aux (Tensor)
    """

    def __init__(self, n_actions: int, hidden_size: int):
        super(FeatureControl, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(hidden_size, 32 * 9 * 9), nn.ReLU(inplace=True)
        )
        self.deconv_value = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2)
        self.deconv_advantage = nn.ConvTranspose2d(
            32, n_actions, kernel_size=4, stride=2
        )

    def forward(self, lstm_feature):
        x = self.fc_layer(lstm_feature).view(-1, 32, 9, 9)
        value = self.deconv_value(x)
        advantage = self.deconv_advantage(x)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_aux = value + advantage - advantage_mean
        return q_aux


class RewardPrediction(nn.Module):
    """
    Modul reward predictions
    Memprediksi reward yang akan didapatkan agen pada unseen frame
    input:
        hasil stack output LSTM terhadap 3 gambar
    output:
        peluang reward (negatif, nol, positif)
    """

    def __init__(
        self,
        hidden_size,
        stack_num=3,
    ):
        super(RewardPrediction, self).__init__()
        self.stack_num = stack_num
        self.prediction_layer = nn.Sequential(
            nn.Linear(hidden_size * stack_num, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, conv_feature):
        probs = self.prediction_layer(conv_feature)
        return F.softmax(probs, dim=1)


class UNREAL(nn.Module):
    """
    Modul Unsupervised Reinforcement and Auxiliary Learning (UNREAL)
    Terdiri dari 4 Komponen:
        Base A3C,
        Replay Buffer,
        Auxiliary Control (Pixel, feature),
        Reward Predictions
    """

    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        device: torch.device,
        hidden_size=256,
        beta=0.01,
        gamma=0.9,
    ):
        super(UNREAL, self).__init__()
        self.beta = beta
        self.gamma = gamma

        self.conv_layer = ConvNet(n_inputs=n_inputs, hidden_size=hidden_size)
        self.lstm_layer = LSTMNet(n_actions=n_actions, hidden_size=hidden_size)
        self.ac_layer = ActorCritic(n_actions=n_actions, hidden_size=hidden_size)
        self.pc_layer = PixelControl(n_actions=n_actions, hidden_size=hidden_size)
        self.rp_layer = RewardPrediction(hidden_size=hidden_size)
        self.to(device)

    def forward(
        self,
        state: torch.Tensor,
        action_oh: torch.Tensor,
        reward: torch.Tensor,
        hidden: Union[tuple, None],
    ):
        conv_feat = self.conv_layer(state)
        lstm_input = torch.cat([conv_feat, action_oh, reward], dim=1)
        lstm_feat, new_hidden = self.lstm_layer(lstm_input, hidden)
        policy, value = self.ac_layer(lstm_feat)
        return policy, value, conv_feat, lstm_feat, new_hidden

    def a3c_loss(
        self,
        states: np.ndarray,
        actions_oh: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        actions: np.ndarray,
    ) -> torch.Tensor:
        states = torch.FloatTensor(states)
        actions_oh = torch.FloatTensor(actions_oh)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        actions = torch.IntTensor(actions)
        
        policy, values, _, _, _ = self.forward(states, actions_oh, rewards)
        dist = Categorical(policy)
        entropy = dist.entropy().unsqueeze(0)
        log_probs = policy.gather(1, actions)

        R = self.calculate_returns(values, dones)
        delta = R.detach() - values
        policy_loss = (-delta.detach() * log_probs - self.beta * entropy).mean()
        value_loss = delta.pow(2).mean()
        return policy_loss + value_loss


    def calculate_returns(self, values, dones):
        R_list = []
        R = 0
        for i in reversed(range(values.size(0))):
            R = R * self.gamma * (1 - dones[i]) + values[i]
            R_list.append(R)
        R_list = list(reversed(R_list))
        return torch.cat(R_list, 0)
