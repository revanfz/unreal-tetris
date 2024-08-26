import random
import numpy as np

from utils import calculate_batch_reward, clip_img
from collections import deque



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, batch_reward=False):
        if batch_reward:
            clipped_obs = clip_img(state, 80)
            clipped_next_obs = clip_img(next_state, 80)
            batch_reward = calculate_batch_reward(clipped_obs, clipped_next_obs)
            self.buffer.append((
                np.expand_dims(state, 0),
                action,
                reward,
                np.expand_dims(next_state, 0),
                batch_reward,
                done
            ))
        else:
            self.buffer.append((
                np.expand_dims(state, 0),
                action,
                reward,
                np.expand_dims(next_state, 0),
                done
            ))

    def sample(self, batch_size, local=False):
        if local:
            state, action, reward, next_state, done = map(np.stack, zip(*self.buffer))
            return state, action, reward, next_state, done
        else:
            batch = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, batch_reward, done = map(np.stack, zip(*batch))
            return state, action, reward, next_state, batch_reward, done
    
    def sample_rp(self, batch_size):
        pass

    def clear(self):
        self.buffer.clear()
        
    def __len__(self):
        return len(self.buffer)