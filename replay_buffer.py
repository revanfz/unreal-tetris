import random
import numpy as np

from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                [
                    np.expand_dims(state, 0),
                    action,
                    reward,
                    np.expand_dims(next_state, 0),
                    done,
                ]
            )
        )

    def sample(self, batch_size):
            max_start_idx = len(self.buffer) - batch_size
            start_idx = random.randint(0, max_start_idx)
            batch = list(self.buffer)[start_idx : start_idx + batch_size]
            state, action, reward, next_state, done = map(
                np.stack, zip(*batch)
            )
            return state, action, reward, next_state, done

    def sample_rp(self, batch_size):
        zero_reward_indices = [
            i for i, (_, _, r, _, _) in enumerate(self.buffer) if r == 0
        ]
        non_zero_reward_indices = [
            i for i, (_, _, r, _, _) in enumerate(self.buffer) if r != 0
        ]

        n_non_zero = min(batch_size // 2, len(non_zero_reward_indices))
        n_zero = batch_size - n_non_zero

        sampled_indices = random.sample(
            non_zero_reward_indices, n_non_zero
        ) + random.sample(zero_reward_indices, n_zero)

        batch = [self.buffer[i] for i in sampled_indices]
        state, _, reward, _, _ = map(np.stack, zip(*batch))
        return state, reward

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
