import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from minibatch import TransitionMemory


class ActorCritic(nn.Module):
    """
    Inisialisasi Model Actor Critic

    Args:
        input_dims = jumlah fitur dari input state
        n_actions = jumlah aksi yang dapat dilakukan agen
        gamma = faktor diskon
        device = perangkat untuk melakuukan komputasi (CPU/GPU)
    """

    def __init__(
        self,
        # input_dims,
        n_actions,
        gamma=0.99,
        beta=0.01,
        ent_coef=0.1,
        lam=0.95,
        device="cuda:0",
    ) -> None:
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.device = device
        self.beta = beta
        self.lam = lam  # lambda
        self.ent_coef = ent_coef  # entropy coefficient
        
        self.memory = TransitionMemory()

        input_layer = [
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            # Output size = ((input size - kernel size) / stride) + 1
            # Output conv pertama = (84 - 8) / 4 + 1 = 76 / 4 = 19 + 1 = 20
            # Output conv kedua = (20 - 4) / 2 + 1 = 16 / 2 = 8 + 1 = 9
            # Karena output 9, maka FC berukuran 9 x 9
        ]

        self.input = nn.Sequential(*input_layer).to(self.device)
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=32 * 9 * 9, out_features=256),
            nn.ReLU(),
        ).to(self.device)
        self.policy = nn.Sequential(
            nn.Linear(in_features=256, out_features=n_actions).to(self.device)
        )
        self.value = nn.Sequential(
            nn.Linear(in_features=256, out_features=1).to(self.device)
        )


    def forward(self, state: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass dari model

        Args:
            state: Sebuah batch vector dari state

        Return:
            policy: tensor dengan action logits (policy), berukuran (1, n_actions)
            value: tensor dengan state value, berukuran (1, 1)
        """
        state = torch.Tensor(state).to(self.device)
        state = self.input(state)
        state = state.reshape(-1,32*9*9)
        state = self.fc_layer(state)
        policy = self.policy(state)
        value = self.value(state)

        return policy, value

    def choose_action(self, observation: np.ndarray) -> torch.Tensor:
        """
        Menentukan aksi yang akan dilakukan oleh agen

        Args:
            observation: Sebuah batch vector dari hasil obersvasi environment

        Returns:
            action: Aksi yang akan dilakukan oleh agen
        """
        self.eval()
        state = torch.tensor(observation, dtype=torch.float32)
        policy, _ = self.forward(state)
        probs = torch.softmax(policy.view(1, -1), dim=1)
        distribution = Categorical(probs)
        action = distribution.sample().detach().cpu().numpy()[0]

        return action

    def calc_R(self, done):
        """
        Menghitung return dari kumpulan step

        Args:
            done: apakah game sudah mencapai terminal state / mencapai t-max
        """
        states = torch.tensor(np.array(self.memory.states), dtype=torch.float32).to(
            self.device
        )
        _, v = self.forward(states)

        R = v[-1] * (1 - int(done))

        batch_return = []
        for reward in self.memory.rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()

        return torch.tensor(batch_return, dtype=torch.float32).to(self.device)

    def calculate_loss(self, done):
        """
        Menghitung loss baru minibatch(transition yang terkumpul dalam satu fase sampling) unutk actor dan critic

        Args:
            done: apakah game sudah mencapai terminal state / mencapai t-max

        Returns:
            total_loss: rata-rata loss dari actor + loss dari critic
        """
        self.train()
        states = torch.tensor(np.array(self.memory.states), dtype=torch.float32).to(
            self.device
        )
        actions = torch.tensor(self.memory.actions, dtype=torch.float32).to(self.device)

        returns = self.calc_R(done)

        # # Melakukan update
        policy, values = self.forward(states)
        values = values.squeeze()
        advantage = returns - values  # td error
        
        probs = torch.softmax(policy, dim=1)
        distribution = Categorical(probs)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        entropy_loss = (self.beta * entropy)
        actor_loss = -((log_probs * advantage.detach()) + entropy_loss)
        critic_loss = advantage.pow(2)

        total_loss = (actor_loss + critic_loss).mean()
        return total_loss
