import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
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
            nn.Linear(self._feature_size(n_inputs[-1], n_inputs[0]), hidden_size),
            nn.ReLU(inplace=True),
        )

    def _feature_size(self, c: int, h: int):
        w = h
        shape = (c, h, w)
        with torch.no_grad():
            o = self.conv_layer(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, observation: torch.Tensor):
        x = self.conv_layer(observation)
        x = x.reshape(x.size(0), -1)
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

    def forward(self, conv_feature: torch.Tensor, hidden: torch.Tensor):
        hx, cx = self.lstm_layer(conv_feature, hidden)
        return hx, cx


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
        q_aux = value + (advantage - advantage_mean)
        q_max = torch.max(q_aux, dim=1, keepdim=False)[0]
        return F.relu(q_aux), q_max


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
        return self.prediction_layer(conv_feature).unsqueeze(0)
        # probs = self.prediction_layer(conv_feature).unsqueeze(0)
        # return F.softmax(probs, dim=1)


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
        gamma=0.99,
        pc: bool = True,
        rp: bool = True,
        vr: bool = True
    ):
        super(UNREAL, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.n_inputs = n_inputs
        self.n_actions = n_actions

        self.use_pc = pc
        self.use_rp = rp
        self.use_vr = vr

        self.conv_layer = ConvNet(n_inputs=n_inputs, hidden_size=hidden_size)
        self.lstm_layer = LSTMNet(n_actions=n_actions, hidden_size=hidden_size)
        self.ac_layer = ActorCritic(n_actions=n_actions, hidden_size=hidden_size)
        if self.use_pc:
            self.pc_layer = PixelControl(n_actions=n_actions, hidden_size=hidden_size)
        if self.use_rp:
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
        lstm_input = torch.cat([conv_feat, action_oh, reward], dim=1).to(self.device)
        hx, cx = self.lstm_layer(lstm_input, hidden)
        policy, value = self.ac_layer(hx)
        return policy, value, hx, cx

    def a3c_loss(
        self,
        R: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        log_probs: torch.Tensor,
        entropies: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        returns = torch.zeros_like(rewards, device=self.device)
        for i in reversed(range(rewards.size(0))):
            R = rewards[i] + self.gamma * dones[i] * R
            returns[i] = R
        
        advantages = returns - values
        value_loss = advantages.pow(2).mean()
        policy_loss = (
            -(log_probs * advantages.detach()).mean() - self.beta * entropies.mean()
        )

        return policy_loss, value_loss

    def control_loss(
        self,
        states: np.ndarray,
        rewards: np.ndarray,
        actions: np.ndarray,
        dones: np.ndarray,
        pixel_changes: np.ndarray,
    ):
        states = torch.Tensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        actions_oh = F.one_hot(actions, self.n_actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        pixel_changes = torch.FloatTensor(np.array(pixel_changes)).to(self.device)

        R = torch.zeros((20, 20), device=self.device)
        conv_feat = self.conv_layer(states[-1].unsqueeze(0))
        lstm_input = torch.cat(
            [conv_feat, actions_oh[-1].unsqueeze(0), rewards[-1].unsqueeze(0)],
            dim=1,
        ).float().to(self.device)
        lstm_feat, _ = self.lstm_layer(lstm_input, None)
        _, R = self.pc_layer(lstm_feat)
        R = R.detach()

        returns = []
        for i in reversed(range(len(rewards[:-1]))):
            R = pixel_changes[i] + 0.9 * ~dones[i] * R
            returns.insert(0, R)
        returns = torch.stack(returns).squeeze(1).to(self.device) # batch 20 20

        conv_feat = self.conv_layer(states[:-1])
        lstm_input = torch.cat([conv_feat, actions_oh[:-1], rewards[:-1]], dim=1).float().to(self.device)
        lstm_feat, _ = self.lstm_layer(lstm_input, None)
        q_aux, _ = self.pc_layer(lstm_feat)  # batch 12 20 20

        pc_a_reshape = actions_oh[:-1].view(-1, self.n_actions, 1, 1)  # batch 12 1 1
        q_taken = (q_aux * pc_a_reshape).sum(dim=1)  # batch 20 20
        pc_loss = F.mse_loss(q_taken, returns)
        return pc_loss

    def rp_loss(self, states: np.ndarray, rewards: np.ndarray):
        actual_reward = rewards[-1]
        # 0 for zero reward
        # 1 for positive reward
        # 2 for negative reward
        if actual_reward == 0:
            reward_class = 0
        elif actual_reward > 0:
            reward_class = 1
        else:
            reward_class = 2

        states = torch.Tensor(np.array(states[:-1])).to(self.device)
        reward_class = torch.tensor(reward_class, dtype=torch.long).unsqueeze(0).to(self.device)

        state_conv_feat = self.conv_layer(states).view(-1)
        reward_prediction = self.rp_layer(state_conv_feat)
        # print("Debug: ", F.softmax(reward_prediction, dim=1).argmax(), reward_class, actual_reward)
        rp_loss = F.cross_entropy(reward_prediction, reward_class)
        return rp_loss

    def vr_loss(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        actions = torch.LongTensor(actions).to(self.device)
        actions_oh = F.one_hot(actions, num_classes=self.n_actions).to(self.device)
        
        with torch.no_grad():
            state = states[-1].unsqueeze(0)
            action = actions_oh[-1].unsqueeze(0)
            reward = rewards[-1].unsqueeze(0)
            _, R, _, _ = self.forward(state, action, reward)
            # R = R.cpu()

        returns = []
        for i in reversed(range(len(rewards[:-1]))):
            R = rewards[i] + self.gamma * (1 - dones[i]) * R
            returns.insert(0, R)
        returns = torch.stack(returns).squeeze(1).to(self.device)

        _, values, _, _ = self.forward(states[:-1], actions_oh[:-1], rewards[:-1])
        loss = F.mse_loss(values, returns)
        return loss


    def batch_pc_loss(self, states, rewards, actions, dones, pixel_changes):
        actions_oh = F.one_hot(actions, self.n_actions).to(self.device)

        R = torch.zeros((20, 20), device=self.device)
        conv_feat = self.conv_layer(states[:, -1])
        lstm_input = torch.cat(
            [conv_feat, actions_oh[:, -1], rewards[:, -1]],
            dim=1,
        ).to(self.device).float()
        lstm_feat, _ = self.lstm_layer(lstm_input, None)
        _, R = self.pc_layer(lstm_feat)
        R = R.detach() * ~dones[:, -1].unsqueeze(-1)

        states = states[:, :-1].reshape(-1, *states.shape[2:])
        actions_oh = actions_oh[:, :-1].reshape(-1, *actions_oh.shape[2:])
        rewards = rewards[:, :-1].reshape(-1, *rewards.shape[2:])

        returns = []
        for i in reversed(range(pixel_changes.size(1) - 1)):
            R = pixel_changes[:, i] + 0.9 * ~dones[:, i].unsqueeze(-1) * R
            returns.insert(0, R)
        returns = torch.stack(returns, dim=1).to(self.device) # batch 20 20
        returns = returns.reshape(-1, *returns.shape[2:])

        conv_feat = self.conv_layer(states)
        lstm_input = torch.cat([conv_feat, actions_oh, rewards], dim=1).float().to(self.device)
        lstm_feat, _ = self.lstm_layer(lstm_input, None)
        q_aux, _ = self.pc_layer(lstm_feat)  # batch 12 20 20

        pc_a_reshape = actions_oh.view(-1, self.n_actions, 1, 1)  # batch 12 1 1
        q_taken = (q_aux * pc_a_reshape).sum(dim=1)  # batch 20 20
        pc_loss = F.mse_loss(q_taken, returns)
        return pc_loss

    def a2c_loss(self, R: torch.Tensor, rewards: torch.Tensor, values, dones, log_probs, entropy):
        returns = torch.zeros_like(rewards)
        for t in reversed(range(rewards.size(0))):
            R = rewards[t] + self.gamma * dones[t] * R
            returns[t] = R
        
        advantages = returns - values
        critic_loss = advantages.pow(2).mean()
        actor_loss = (
            -(advantages.detach() - log_probs).mean() - self.beta * entropy.mean()
        )

        return (actor_loss, critic_loss)
    
    def batch_rp_loss(self, states: torch.Tensor, rewards: torch.Tensor):
        actual_reward = rewards[:, -1]
        states = states[:, :-1].reshape(-1, *states.shape[2:])

        # states = states[:, :-1].reshape(-1, *states.shape[2:])
        # 0 for zero reward
        # 1 for positive reward
        # 2 for negative reward
        reward_class = torch.where(actual_reward > 0, 1, torch.where(actual_reward < 0, 2, 0))
        
        state_conv_feat = self.conv_layer(states).view(rewards.size(0), -1)
        reward_prediction = self.rp_layer(state_conv_feat).squeeze()
        rp_loss = F.cross_entropy(reward_prediction, reward_class.squeeze(1))
        return rp_loss
    
    def batch_vr_loss(self, states: torch.Tensor, rewards: torch.Tensor, actions: torch.Tensor, dones: torch.Tensor):
        actions_oh = F.one_hot(actions, self.n_actions).to(self.device)

        with torch.no_grad():
            _, R, _, _ = self.forward(
                states[:, -1], actions_oh[:, -1], rewards[:, -1]
            )
            R = R * ~dones[:, -1]

        states = states[:, :-1]
        rewards = rewards[:, :-1]
        actions_oh = actions_oh[:, :-1]

        returns = []
        for i in reversed(range(rewards.size(1))):
            R = rewards[:, i] + self.gamma * ~dones[:, i] * R
            returns.insert(0, R)
        returns = torch.stack(returns, dim=1).float().to(self.device) # batch 20 20

        returns = returns.reshape(-1, *returns.shape[2:])
        states = states.reshape(-1, *states.shape[2:])
        rewards = rewards.reshape(-1, *rewards.shape[2:])
        actions_oh = actions_oh.reshape(-1, *actions_oh.shape[2:])

        _, values, _, _ = self.forward(
            states, actions_oh, rewards
        )
        values = values.float()
        vr_loss = F.mse_loss(values, returns)
        return vr_loss
