import torch.nn as nn

from torch.cuda import is_available
from torch import Tensor, zeros


class ActorCritic(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int) -> None:
        super(ActorCritic, self).__init__()
        self.device = "cuda" if is_available() else "cpu"
        input_layers = [
            nn.Conv2d(num_inputs, 16, 8, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=0),
        ]
        self.convolutional = nn.Sequential(*input_layers)
        self.fc = nn.Sequential(nn.Linear(32 * 10 * 10, 256), nn.ReLU(inplace=True))
        self.critic = nn.Linear(256, 1)
        self.actor = nn.Linear(256, num_actions)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.convolutional(x).unsqueeze(0)
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
        R = values[-1] * int(done)

        for t in reversed(range(T)):
            R = rewards[t] + gamma * R
            td_errors[t] = R

        advantage = td_errors - values
        critic_loss = advantage.pow(2).mean()
        actor_loss = -(log_probs * advantage.detach()).mean() - beta * entropies.mean()
        total_loss = actor_loss + 0.5 * critic_loss

        return total_loss
