import sys
import pygame
import tetris_game.tetris.tetris as tetris
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}

    def __init__(self, render_mode=None):
        self.pygame = tetris.Tetris()

        # Aksi yang bisa dilakukan: kiri, kanan, rotasi, speedup
        self.action_space = Discrete(4)

        # Array matriks arena tetris
        self.observation_space = Dict({
            "board": Box(low=0, high=1, shape=(tetris.ROW, tetris.COL), dtype=np.uint8)
        })

        self.max_episode = 200

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        del self.pygame
        self.pygame = tetris.Tetris()
        obs = self.pygame._get_obs()
        info = self.pygame._get_info()
        return obs, info

    def step(self, action):
        self.pygame.step(action)
        obs = self.pygame._get_obs()
        reward = self.pygame.evaluate()
        done = self.pygame.is_done()
        info = self.pygame._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, done, False, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()

    def _render_frame(self):
        try:
            if self.render_mode == 'human':
                self.pygame.view()
                self.pygame.clock.tick(self.metadata["render_fps"])
        except KeyboardInterrupt:
            print("Program Dihentikan")
            sys.exit()

    def close(self):
        pygame.display.quit()
        pygame.quit()
