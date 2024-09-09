import random
import numpy as np

from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.non_rewarding_indices = deque()
        self.rewarding_indices = deque()
        self.highest_indices = 0

    def store(
        self,
        state: np.ndarray,
        prev_action: int,
        prev_reward: float,
        next_state: np.ndarray,
        next_action: int,
        next_reward: float,
        done: bool,
    ):
        frame_index = len(self.buffer) + self.highest_indices

        self.buffer.append(
            (
                [
                    np.expand_dims(state, 0),
                    prev_action,
                    prev_reward,
                    np.expand_dims(next_state, 0),
                    next_action,
                    next_reward,
                    done,
                ]
            )
        )

        if frame_index >= 3:
            if next_reward != 0:
                self.rewarding_indices.append(frame_index)
            else:
                self.non_rewarding_indices.append(frame_index)

        if self._is_full():
            self.highest_indices += 1
            cut_frame_index = self.highest_indices + 2

            while len(self.non_rewarding_indices) > 0 and self.non_rewarding_indices[0] < cut_frame_index:
                self.non_rewarding_indices.popleft()

            while len(self.rewarding_indices) > 0 and self.rewarding_indices[0] < cut_frame_index:
                self.rewarding_indices.popleft()

    def _is_full(self):
        return len(self.buffer) >= self.buffer.maxlen

    def sample(self, batch_size, base = False):
        max_start_idx = len(self.buffer) - batch_size
        start_idx = random.randint(0, max_start_idx)
        if base:
            start_idx = max_start_idx
        batch = list(self.buffer)[start_idx : start_idx + batch_size]
        (
            states,
            prev_actions,
            prev_rewards,
            next_states,
            next_actions,
            next_rewards,
            dones,
        ) = map(np.stack, zip(*batch))
        return (
            states,
            prev_actions,
            prev_rewards,
            next_states,
            next_actions,
            next_rewards,
            dones,
        )

    def sample_rp(self, batch_size=3):
        if random.random() < 0.5:
            zero_reward = True
        else:
            zero_reward = False

        if len(self.rewarding_indices) < 1:
            zero_reward = True
        elif len(self.non_rewarding_indices) < 1:
            zero_reward = False

        if zero_reward:
            index = np.random.randint(0, len(self.non_rewarding_indices))
            end_frame_index = self.non_rewarding_indices[index]
        else:
            index = np.random.randint(0, len(self.rewarding_indices))
            end_frame_index = self.rewarding_indices[index]

        start_frame_index = end_frame_index - (batch_size - 1)
        raw_start_frame_index = start_frame_index - self.highest_indices

        batch = [self.buffer[raw_start_frame_index + i] for i in range(batch_size)]
        (
            states,
            prev_actions,
            prev_rewards,
            next_states,
            next_actions,
            next_rewards,
            dones,
        ) = map(np.stack, zip(*batch))
        return states, prev_rewards, next_rewards

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
