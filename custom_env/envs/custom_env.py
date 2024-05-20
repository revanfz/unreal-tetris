import os
import random
import sys
import pygame
import numpy as np
import gymnasium as gym

from custom_env import game

sys.path.append("/")

from random import choice
from gymnasium.spaces import Box, Dict, Discrete

from custom_env.game.matrix import Matrix
from custom_env.game.settings import (
    MATRIX_WIDTH,
    MATRIX_HEIGHT,
    TETROMINOS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    PIXEL,
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
                "matrix_image": Box(
                    low=0,
                    high=255,
                    shape=(MATRIX_HEIGHT, MATRIX_WIDTH, 3),
                    dtype=np.uint8,
                )
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
                    if (
                        not self.game.field_data[row][col]
                        and self.game.field_data[row - 1][col]
                    ):
                        holes += 1
            holes_per_col.append(holes)

        return holes_per_col

    def count_transitions(self):
        col_transition = 0
        for col in range(len(self.game.field_data[0])):
            for row in range(len(self.game.field_data) - 1):
                if row == len(self.game.field_data):
                    break
                elif bool(self.game.field_data[row + 1][col]) ^ bool(
                    self.game.field_data[row][col]
                ):
                    col_transition += 1

        row_transition = 0
        for row in range(len(self.game.field_data)):
            for col in range(len(self.game.field_data[0]) - 1):
                if col == len(self.game.field_data[0]):
                    break
                elif bool(self.game.field_data[row][col + 1]) ^ bool(
                    self.game.field_data[row][col]
                ):
                    row_transition += 1

        return (row_transition, col_transition)

    def count_wells(self):
        wells = 0
        for col in range(len(self.game.field_data[0]) - 1):
            for row in range(len(self.game.field_data)):
                cell = self.game.field_data[row][col]
                if col == 0:
                    if not cell and self.game.field_data[row][col + 1]:
                        wells += 1
                elif col == range(len(self.game.field_data[0]) - 1):
                    if not cell and self.game.field_data[row][col - 1]:
                        wells += 1
                else:
                    if (
                        not cell
                        and self.game.field_data[row][col + 1]
                        and self.game.field_data[row][col - 1]
                    ):
                        wells += 1

        return wells

    def _get_info(self):
        col_heights = self.find_col_heights()
        holes = self.calculate_holes(col_heights)
        lines_cleared = self.game.last_deleted_rows
        row_transition, col_transition = self.count_transitions()
        cumulated_wells = self.count_wells()

        return {
            "lines_cleared": lines_cleared,
            "row_transitions": row_transition,
            "col_transitions": col_transition,
            "holes": sum(holes),
            "cumulative_wells": cumulated_wells,
            "score": self.game.current_scores,
            "total_lines": self.game.current_lines
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

        self.render()

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

        self.render()

        info = self._get_info()
        observation = self._get_obs()

        reward = self.evaluate(info)

        return observation, reward, self.game.tetromino.game_over, False, info

    def evaluate(self, info):
        reward = (
            -4 * info["holes"]
            - info["cumulative_wells"]
            - info["row_transitions"]
            - info["col_transitions"]
            - self.game.last_landing_height
        )
        if self.game.last_block_placed != self.game.block_placed:
            reward += 10
            self.game.last_block_placed = self.game.block_placed
        if info["lines_cleared"] > 0:
            reward += CLEAR_REWARDS[info["lines_cleared"]] ** 2
        if self.game.block_placed % 10 == 0 and self.game.block_placed > 0:
            reward += 5
        return reward

    def render(self):
        if self.render_mode == "rgb_array" or self.render_mode == "human":
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
                sfc = self.game.run(canvas)
                self.score.run(canvas)
                self.preview.run(self.next_shapes, canvas)

                matrix_screen = pygame.transform.rotate(canvas.subsurface(
                    pygame.Rect(PIXEL, PIXEL, MATRIX_WIDTH, MATRIX_HEIGHT)
                ), 90)

                self.matrix_screen_array = pygame.surfarray.array3d(matrix_screen)

                self.game.draw_line(canvas, sfc)

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
        except KeyboardInterrupt:
            print("Closing the window...")
            pygame.quit()
            sys.exit()


if __name__ == "__main__":
    raise NotImplementedError("This script do not supposed to be run directly")
