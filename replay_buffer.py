import numpy as np

from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.non_rewarding_indices = deque()
        self.rewarding_indices = deque()
        self.highest_index = 0

    def store(
        self,
        state: np.ndarray,
        reward: float,
        action: int,
        done: bool,
        pixel_change: float,
    ):
        if done and len(self.buffer) > 0 and self.buffer[-1][3]:
            print("Discarding consecutive terminal frame")
            return

        frame_index = self.highest_index + len(self.buffer)
        is_full = self._is_full()

        self.buffer.append([state, reward, action, done, pixel_change])

        if frame_index >= 3:
            if reward == 0:
                self.non_rewarding_indices.append(frame_index)
            else:
                self.rewarding_indices.append(frame_index)

        if is_full:
            self.highest_index += 1
            cut_frame_index = self.highest_index + 3

            if (
                len(self.non_rewarding_indices) > 0
                and self.non_rewarding_indices[0] < cut_frame_index
            ):
                self.non_rewarding_indices.popleft()

            if (
                len(self.rewarding_indices) > 0
                and self.rewarding_indices[0] < cut_frame_index
            ):
                self.rewarding_indices.popleft()

    def sample(self, size):
        start_pos = len(self.buffer) - size
        sampled_frames = []
        for i in range(size):
            frame = self.buffer[start_pos + i]
            sampled_frames.append(frame)

        return map(list, zip(*sampled_frames))

    def sample_sequence(self, size):
        start_pos = np.random.randint(0, len(self.buffer) - size - 1)
        if self.buffer[start_pos][3]:
            start_pos += 1

        sampled_frames = []

        for i in range(size):
            frame = self.buffer[start_pos + i]
            sampled_frames.append(frame)
            if frame[3]:
                break

        return map(list, zip(*sampled_frames))

    def sample_rp(self):
        if np.random.randint(2) == 0:
            from_zero = True
        else:
            from_zero = False

        if len(self.rewarding_indices) == 0:
            from_zero = True
        elif len(self.non_rewarding_indices) == 0:
            from_zero = False

        if from_zero:
            index = np.random.randint(0, len(self.non_rewarding_indices))
            end_frame_index = self.non_rewarding_indices[index]
        else:
            index = np.random.randint(0, len(self.rewarding_indices))
            end_frame_index = self.rewarding_indices[index]

        start_frame_index = end_frame_index - 3
        raw_start_frame_index = start_frame_index - self.highest_index

        sampled_frames = []

        for i in range(4):
            frame = self.buffer[raw_start_frame_index + i]
            sampled_frames.append(frame)

        return map(list, zip(*sampled_frames))

    def _is_full(self):
        return len(self.buffer) >= self.buffer.maxlen

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
