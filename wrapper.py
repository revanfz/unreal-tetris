import os
import time
import numpy as np
import gymnasium as gym

from gymnasium import error
from moviepy.video.fx.resize import resize
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

LINE_REWARDS = {1: 40, 2: 100, 3: 300, 4: 1200}
LINE_REWARDS = {1: 40, 2: 100, 3: 300, 4: 1200}


class FrameSkipWrapper(gym.Wrapper):
    def __init__(
        self, env, skip=4,
    ):
        super().__init__(env)
        self.env = env
        self.skip = skip
        self.lines = 0
        self.blocks = 1
        self.fitness = 0
        self.prev_action = 0

    def step(self, action):
        done = False
        total_rewards = 0.0
        rotation_reward = 0.0
        for i in range(self.skip):
            if i == 0 and np.random.rand() < (1/self.skip):
                action_taken = self.prev_action
            else:
                action_taken = action
            obs, reward, done, truncated, info = self.env.step(action_taken)
            total_rewards += reward
            if action in [1, 2, 4, 5, 7, 8, 10, 11]:
                if info['current_piece'] != 'O':
                    rotation_reward += 0.5
                else:
                    total_rewards -= 0.1
            if done:
            #     total_rewards -= 20
                break
        
        blocks = sum(info["statistics"].values())
        if self.blocks < blocks:
            total_rewards += self.reward_func() + rotation_reward
            self.blocks = blocks
        lines = info['number_of_lines']
        if self.lines < lines:
            lines_cleared = lines - self.lines
            total_rewards += LINE_REWARDS[lines_cleared] + 0.76 * lines_cleared
            self.lines += lines_cleared
        self.prev_action = action_taken
        return obs, total_rewards, done, truncated, info
    

    def reset(self, seed=None, options=None):
        self.lines = 0
        self.blocks = 1
        self.fitness = 0
        self.prev_action = 0
        return self.env.reset(seed=seed, options=options)

    def uneven_penalty(self, board: np.ndarray):
        board_height = (20 - np.argmax(board != 0, axis=0)) * (board.any(axis=0))
        bumpiness = np.abs(np.diff(board_height))
        height_penalty = self.env.unwrapped._board_height * -2.51
        bumpiness_penalty = bumpiness.sum() * -0.18
        return height_penalty, bumpiness_penalty

    def hole_penalty(self, board):
        filled_above = np.maximum.accumulate(board != 0, axis=0)
        holes = np.sum((filled_above & (board == 0)))
        return holes * -0.36
   
    def fitness_function(self):
        board = self.env.unwrapped._board
        board[board == 239] = 0
        hp, bp = self.uneven_penalty(board)  # height penalty, bumpiness penalty
        hole_penalty = self.hole_penalty(board)
        # print(f"Height penalty {hp}, Bumpiness penalty  {bp}, Hole penalty {hole_penalty}")
        return hp + hole_penalty + bp
    
    def reward_func(self):
        fitness_value = self.fitness_function()
        reward = fitness_value - self.fitness
        self.fitness = fitness_value
        return reward



class RecordVideo(gym.Wrapper):
    def __init__(self, env, path: str, format: str):
        super().__init__(env)
        self.env = env
        self.path = path
        self.format = format
        self.episode = 1
        self.frame_captured = []

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.frame_captured.append(self.env.render().copy())
        if done:
            self.close()
            self.episode += 1
        return obs, reward, done, truncated, info

    def close(self):
        if len(self.frame_captured) > 0:
            if self.format in ["mp4", "avi", "webm", "ogv", "gif"]:
                filename = "{}/{}.{}".format(self.path, self.episode, self.format)
                clip = ImageSequenceClip(
                    self.frame_captured, fps=self.env.metadata.get("fps", 60)
                ).fx(resize, width=480)
                if self.format == "gif":
                    clip.write_gif(filename)
                else:
                    clip.write_videofile(filename, threads=2)
            else:
                raise error.Error(
                    f"Invalid recording format. Supported are mp4, avi, webm, ogv, gif"
                )
        self.frame_captured.clear()
