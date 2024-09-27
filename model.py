from typing import Union

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from utils import preprocessing


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

    def __init__(self, n_inputs: tuple, hidden_size: int):
        super(ConvNet, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(n_inputs[-1], 16, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self._feature_size(n_inputs[-1], n_inputs[0]), hidden_size), nn.ReLU(inplace=True)
        )

    def _feature_size(self, c: int, h: int):
        w = h
        shape = (c, h, w)
        with torch.no_grad():
            o = self.conv_layer(torch.zeros(1, *shape))
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
        self.lstm_layer = nn.LSTMCell(hidden_size + n_actions + 1, hidden_size)

    def forward(self, conv_feature, hidden):
        lstm_feature, new_hidden = self.lstm_layer(conv_feature, hidden)
        return lstm_feature, new_hidden


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
        self.deconv_spatial = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3), nn.ReLU(inplace=True)
        )
        self.deconv_value = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2)
        self.deconv_advantage = nn.ConvTranspose2d(
            32, n_actions, kernel_size=4, stride=2
        )

    def forward(self, lstm_feature: torch.Tensor):
        x = self.fc_layer(lstm_feature).view(-1, 32, 7, 7)
        spatial_feat = self.deconv_spatial(x)
        value = self.deconv_value(spatial_feat)
        advantage = self.deconv_advantage(spatial_feat)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_aux = value + advantage - advantage_mean
        return F.relu(q_aux)


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
        probs = self.prediction_layer(conv_feature).unsqueeze(0)
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
        self.device = device
        self.n_actions = n_actions

        self.conv_layer = ConvNet(n_inputs=n_inputs, hidden_size=hidden_size)
        self.lstm_layer = LSTMNet(n_actions=n_actions, hidden_size=hidden_size)
        self.ac_layer = ActorCritic(n_actions=n_actions, hidden_size=hidden_size)
        self.pc_layer = PixelControl(n_actions=n_actions, hidden_size=hidden_size)
        self.rp_layer = RewardPrediction(hidden_size=hidden_size)
        self.to(self.device)

    def forward(
        self,
        state: torch.Tensor,
        action_oh: torch.Tensor,
        reward: torch.Tensor,
        hidden: Union[tuple, None] = None,
    ):
        conv_feat = self.conv_layer(state)
        lstm_input = torch.cat([conv_feat, action_oh, reward], dim=1)
        lstm_feat, lstm_cell = self.lstm_layer(lstm_input, hidden)
        policy, value = self.ac_layer(lstm_feat)
        return policy, value, conv_feat, lstm_feat, lstm_cell

    def a3c_loss(
        self,
        states: np.ndarray,
        rewards: np.ndarray,
        actions: np.ndarray,
        dones: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = torch.FloatTensor(states).to(self.device).squeeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        actions = torch.LongTensor(actions).to(self.device)
        actions_oh = F.one_hot(actions, num_classes=self.n_actions).to(self.device)

        policy, values, _, _, _ = self.forward(states, actions_oh, rewards)
        dist = Categorical(policy)
        entropy = dist.entropy().unsqueeze(0)
        log_probs = dist.log_prob(actions)

        R = self.calculate_returns(rewards, dones, values[-1]).unsqueeze(1)
        advantage = R.detach() - values
        policy_loss = (-advantage.detach() * log_probs - self.beta * entropy).mean()
        value_loss = advantage.pow(2).mean()
        return (policy_loss, value_loss, entropy)

    def control_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        next_actions: np.ndarray,
        next_rewards: np.ndarray,
        dones: np.ndarray,
    ):
        states = torch.FloatTensor(states).to(self.device).squeeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device).squeeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        actions = torch.LongTensor(actions).to(self.device)
        actions_oh = F.one_hot(actions, num_classes=self.n_actions).to(self.device)
        next_actions = torch.LongTensor(next_actions).to(self.device)
        next_actions_oh = F.one_hot(next_actions, num_classes=self.n_actions).to(self.device)
        next_rewards = torch.FloatTensor(next_rewards).to(self.device).unsqueeze(1)

        conv_feat = self.conv_layer(states)
        lstm_input = torch.cat([conv_feat, actions_oh, rewards], dim=1)
        lstm_feat, _ = self.lstm_layer(lstm_input, None)
        q_aux: torch.Tensor = self.pc_layer(lstm_feat)

        next_conv_feat = self.conv_layer(next_states)
        next_lstm_input = torch.cat([next_conv_feat, next_actions_oh, next_rewards], dim=1)
        next_lstm_feat, _ = self.lstm_layer(next_lstm_input, None)
        next_q_aux: torch.Tensor = self.pc_layer(next_lstm_feat)

        cropped_states = preprocessing(states, pixel_control=True)
        cropped_next_states = preprocessing(next_states, pixel_control=True)
        pixel_change = torch.abs(cropped_next_states - cropped_states)
        pixel_change = F.avg_pool2d(pixel_change, kernel_size=4, stride=4)
        reward = pixel_change.mean(dim=1).unsqueeze(1)
        
        actions = actions.view(actions.size(0), 1, 1, 1)
        actions = actions.expand(-1, -1, q_aux.size(2), q_aux.size(3))
        q_values = q_aux.gather(1, actions).squeeze(0)
        next_q = next_q_aux.max(1)[0].detach().unsqueeze(1)
        dones = dones.view(-1, 1, 1, 1).expand_as(reward)
        q_target = reward * self.gamma * (1 - dones) * next_q
        pc_loss = F.mse_loss(q_values, q_target)

        return pc_loss

    
    def rp_loss(
        self,
        states: np.ndarray,
        rewards: np.ndarray,
        next_rewards: float
    ):
        states = torch.FloatTensor(states).to(self.device).squeeze(1)
        reward = next_rewards
        rewards = torch.zeros(3, dtype=torch.float32).to(self.device)
        if reward > 0:
            rewards[2] = 1.0
        elif reward < 0:
            rewards[1] = 1.0
        else:
            rewards[0] = 1.0
            
        state_conv_feat = self.conv_layer(states).view(-1)
        reward_classification = self.rp_layer(state_conv_feat)
        rp_loss = F.cross_entropy(reward_classification, rewards.unsqueeze(0))

        return rp_loss
    
    def vr_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray
    ):
        states = torch.FloatTensor(states).to(self.device).squeeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        actions = torch.LongTensor(actions).to(self.device)
        actions_oh = F.one_hot(actions, num_classes=self.n_actions).to(self.device)

        _, values, _, _, _ = self.forward(states, actions_oh, rewards)
        R = self.calculate_returns(rewards, dones, values[-1]).unsqueeze(1)
        loss = (R.detach() - values).pow(2).mean()
        return loss    

    def calculate_returns(self, rewards, dones, last_value):
        R_list = []
        R = last_value
        for i in reversed(range(rewards.size(0))):
            R = R * self.gamma * (1 - dones[i]) + rewards[i]
            R_list.append(R)
        R_list = list(reversed(R_list))
        return torch.cat(R_list, 0)
