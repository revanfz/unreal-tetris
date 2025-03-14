import os
import numpy as np
import gymnasium as gym

from gymnasium import error
from moviepy.video.fx.resize import resize
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

LINE_REWARDS = {1: 40, 2: 100, 3: 300, 4: 1200}


class FrameSkipWrapper(gym.Wrapper):
    def __init__(
        self, env, skip=4, level=19
    ):
        super().__init__(env)
        self.env = env
        self.skip = skip
        self.lines = 0
        self.lines_cleared = 0
        self.blocks = 1
        self.fitness = 0
        self.level = level
        self.lines_history = []

    def step(self, action):
        total_rewards = 0.0
        for step in range(self.skip):
            if step > 0 and action != 5:
                action = 0
            obs, _, done, truncated, info = self.env.step(action)
            total_rewards += self.reward_func()
            if done:
                total_rewards -= 100
                break
        
        return obs, total_rewards, done, truncated, info

    def process_board(self):
        board_copy = self.env.unwrapped._board
        board_copy[board_copy == 239] = 0
        board_copy[board_copy != 0] = 1
        return board_copy
    
    def reset(self, seed=None, options=None):
        self.lines = 0
        self.blocks = 1
        self.fitness = 0
        self.lines_history.clear()
        return self.env.reset(seed=seed, options=options)

    def uneven_penalty(self, board: np.ndarray):
        board_height = (20 - np.argmax(board != 0, axis=0)) * (board.any(axis=0))
        bumpiness = np.abs(np.diff(board_height))
        height_penalty = board_height.sum() * -0.51
        bumpiness_penalty = bumpiness.sum() * -0.18
        return height_penalty, bumpiness_penalty

    def hole_penalty(self, board):
        filled_above = np.maximum.accumulate(board != 0, axis=0)
        holes = np.sum((filled_above & (board == 0)))
        return holes * -0.36
    
    def line_reward(self, board):
        lines_cleared = np.sum(np.all(board == 1, axis=1))
        if lines_cleared > self.lines_cleared:
            self.lines += lines_cleared
            self.lines_history.append(lines_cleared)
        self.lines_cleared = lines_cleared
        return self.lines * 0.76
   
    def fitness_function(self, board):
        hp, bp = self.uneven_penalty(board)  # height penalty, bumpiness penalty
        hole_penalty = self.hole_penalty(board)
        lines_r = self.line_reward(board)
        return hp + lines_r + hole_penalty + bp
    
    def reward_func(self):
        board = self.process_board()
        fitness_value = self.fitness_function(board)
        reward = fitness_value - self.fitness
        self.fitness = fitness_value
        return round(reward, 2)


class RecordVideo(gym.Wrapper):
    def __init__(self, env, path: str, format: str, log_every: int = 100, episode = 0, recording = True):
        super().__init__(env)
        self.env = env
        self.path = path
        self.format = format
        self.episode = episode
        self.log_every = log_every
        self.frame_captured = []
        self.recording = recording

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.frame_captured.append(self.env.unwrapped.screen.copy())
        if done:
            self.record()
            if self.recording:
                self.episode += 1
        return obs, reward, done, truncated, info

    def record(self):
        if len(self.frame_captured) > 0 and self.recording:
            if self.format in ["mp4", "avi", "webm", "ogv", "gif"]:
                filename = "{}/{}.{}".format(self.path, int(self.episode // self.log_every + 1), self.format)
                clip = ImageSequenceClip(
                    self.frame_captured[::2], fps=self.env.metadata.get("fps", 30)
                ).fx(resize, width=480)
                # ).with_effects([Resize(width=480)])
                if self.episode % self.log_every == 0:
                    if self.format == "gif":
                        clip.write_gif(filename)
                    else:
                        clip.write_videofile(filename, logger=None)
            else:
                raise error.Error(
                    f"Invalid recording format. Supported are mp4, avi, webm, ogv, gif"
                )

    def reset(self, seed=None, options=None):
        self.frame_captured.clear()
        return self.env.reset(seed=seed, options=options)