import os
import random
import sys
import pygame
import numpy as np
import gymnasium as gym

sys.path.append("/")

from random import choice
from gymnasium.spaces import Box, Dict, Discrete
from stable_baselines3.common.env_checker import check_env

from custom_env.game.matrix import Matrix
from custom_env.game.settings import (
    MATRIX_WIDTH,
    MATRIX_HEIGHT,
    TETROMINOS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    SIDEBAR_WIDTH,
    PREVIEW_HEIGHT,
    PIXEL,
    ROW,
    COL,
    IMG_DIR,
    CLEAR_REWARDS,
)
from custom_env.game.preview import Preview
from custom_env.game.score import Score


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("Tetris Smart Agent")

        self.window_size = (WINDOW_WIDTH, WINDOW_HEIGHT)

        self.observation_space = Dict(
            {
                "matrix_image": Box(low=0, high=255, shape=(MATRIX_WIDTH + PIXEL * 2, MATRIX_HEIGHT + PIXEL * 2, 3), dtype=np.uint8),
                # "lines_cleared": Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                # "holes": Box(low=0, high=9, shape=(COL,)),
                # "quadratic_unevenness": Box(low=0, high=981, shape=(1,), dtype=np.float32),
                # "sum_height": Box(low=0, high=100, shape=(1,), dtype=np.float32),
                # "current_shape": Box(low=1, high=7, shape=(1,), dtype=np.float32),
                # "next_shape": Box(low=1, high=7, shape=(3,), dtype=np.float32),
            }
        )
        self.action_space = Discrete(7)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.game_over_screen = None

    def get_next_shape(self):
        if self.game.block_placed % 7 == 0 and self.game.block_placed > 3:
            random.shuffle(self.bag)

        next_shape = self.next_shapes.pop(0)
        # self.next_shapes.append('Z')
        self.next_shapes.append(
            self.bag[
                (
                    self.game.block_placed % 7
                    if self.game.block_placed > 3
                    else self.game.block_placed + 3
                )
            ]
        )

        return next_shape

    def update_score(self, lines, scores, level):
        self.score.lines = lines
        self.score.level = level
        self.score.scores = scores

    def find_col_heights(self):
        max_rows = []
        for col in range(len(self.game.field_data[0])):
            max_row = 20
            for row in range(len(self.game.field_data) - 1, -1, -1):
                if self.game.field_data[row][col]:
                    max_row = row
            max_row = 20 - max_row
            max_rows.append(max_row)

        return max_rows

    def calculate_holes(self, max_rows):
        holes_per_col = []
        for col, max_row in enumerate(max_rows):
            holes = 0
            if max_row != 0:
                for row in range(len(self.game.field_data) - 1, 20 - max_row, -1):
                    if not self.game.field_data[row][col]:
                        holes += 1
            holes_per_col.append(holes)

        return holes_per_col

    def calculate_unevenness(self, max_rows):
        unevenness = 0
        for i in range(len(max_rows) - 1):
            # beda_absolute = abs(max_rows[i + 1] - max_rows[i])
            beda_absolute = (max_rows[i + 1] - max_rows[i]) ** 2
            unevenness += beda_absolute

        return unevenness

    def _get_info(self):
        col_heights = self.find_col_heights()
        sum_height = sum(col_heights)
        holes = self.calculate_holes(col_heights)
        unevenness = self.calculate_unevenness(col_heights)
        lines_cleared = self.game.last_deleted_rows

        return {
            "lines_cleared": lines_cleared,
            "sum_height": sum_height,
            "holes": sum(holes),
            "quadratic_unevenness": unevenness,
        }

    def _get_obs(self):
        return {
            "matrix_image": self.matrix_screen_array,
            # "preview_image": self.preview_screen_array,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.bag = ["Z", "S", "O", "L", "J", "I", "T"]
        random.shuffle(self.bag)

        self.next_shapes = self.bag[:4]

        self.game = Matrix(
            self.get_next_shape,
            self.update_score,
            initial_shape=self.next_shapes.pop(0),
        )
        self.score = Score()
        self.preview = Preview()

        if self.render_mode == "human":
            self._render_frame()
        
        info = self._get_info()
        observation = self._get_obs()

        return observation, info

    def step(self, action):
        if action == 0:  # Noop (No-operation)
            pass

        # if not self.game.timers["rotate"].active:
        if action == 1 or action == 2:  # rotation
            self.game.tetromino.rotate("right" if action == 1 else "left")
            # self.game.timers["rotate"].activate()

        # if not self.game.timers["horizontalMove"].active: # kanan
        if action == 3:
            self.game.input(1)
            # self.game.timers["horizotalMove"].activate()

        if action == 4:  # kiri
            self.game.input(-1)
            # self.game.timers["horizontalMove"].activate()

        if action == 5:
            self.game.soft_drop()

        if action == 6:  # drop
            self.game.drop()

                
        if self.render_mode == "human":
            self._render_frame()

        info = self._get_info()
        observation = self._get_obs()

        reward = self.evaluate(info)

        return observation, reward, self.game.tetromino.game_over, False, info

    def evaluate(self, info):
        reward = self.game.block_placed
        reward -= 0.01 * info["sum_height"]
        reward -= info["holes"] * 0.5 - info["quadratic_unevenness"] * 0.5
        if self.game.tetromino.game_over:
            reward = -100
        else:
            if self.game.last_block_placed != self.game.block_placed:
                reward += 10
                if self.game.last_deleted_rows > 0:
                    reward += CLEAR_REWARDS[self.game.last_deleted_rows] ** 2
                self.game.last_block_placed = self.game.block_placed

        return reward

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        try:
            if self.window is None and self.render_mode == "human":
                self.window = pygame.display.set_mode(self.window_size)
            if self.clock is None and self.render_mode == "human":
                self.clock = pygame.time.Clock()

            canvas = pygame.Surface(self.window_size)

            if not self.game.tetromino.game_over:
                canvas.fill((67, 70, 75))
                self.game.run(canvas)
                self.score.run(canvas)
                self.preview.run(self.next_shapes, canvas, is_training=False)
            else:
                if self.game_over_screen is None:
                    self.game_over_screen = os.path.join(IMG_DIR, "game_over.jpg")
                    pygame.image.save(canvas, self.game_over_screen)
                    self.game_over_screen = pygame.image.load(self.game_over_screen)
                canvas.blit(self.game_over_screen, (0, 0))

            if self.render_mode == "human":
                self.window.blit(canvas, canvas.get_rect())

                pygame.event.pump()
                pygame.display.update()
                self.clock.tick(self.metadata["render_fps"])

                # if self.game.last_block_placed != self.game.block_placed:
                matrix_screen = canvas.subsurface(pygame.Rect(0, 0, MATRIX_WIDTH + PIXEL * 2, MATRIX_HEIGHT + PIXEL * 2))
                preview_screen = canvas.subsurface(pygame.Rect(MATRIX_WIDTH + PIXEL, 0, SIDEBAR_WIDTH + PIXEL*2, PREVIEW_HEIGHT * WINDOW_HEIGHT + PIXEL))

                self.matrix_screen_array = pygame.surfarray.array3d(matrix_screen)
                self.preview_screen_array = pygame.surfarray.array3d(preview_screen)
                
                return (self.matrix_screen_array, self.preview_screen_array)
            else:  # rgb_array
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
                )
        except KeyboardInterrupt:
            print("Closing the window...")
            pygame.quit()
            sys.exit()


if __name__ == "__main__":
    raise NotImplementedError("This script do not supposed to be run directly")
