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
        self.action_space = Discrete(13)

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
                    if col == 0:
                        left_side = True
                    else:
                        left_side = bool(self.game.field_data[row][col-1])
                    if col == len(max_rows) - 1:
                        right_side = True
                    else:
                        right_side = bool(self.game.field_data[row][col+1])
                    if (
                        self.game.field_data[row - 1][col]
                        and not self.game.field_data[row][col]
                        and not left_side
                        and not right_side
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
    
    def calculate_bump(self, col_heights):
        bumpiness = 0
        for i in range(len(col_heights)-1):
            bumpiness += abs(col_heights[i] - col_heights[i + 1])

        return bumpiness 

    def _get_info(self):
        col_heights = self.find_col_heights()
        holes = self.calculate_holes(col_heights)
        lines_cleared = self.game.last_deleted_rows
        bumpiness = self.calculate_bump(col_heights)
        # row_transition, col_transition = self.count_transitions()
        # cumulated_wells = self.count_wells()

        return {
            "heights": sum(col_heights),
            "lines_cleared": lines_cleared,
            "holes": sum(holes),
            "bumpiness": bumpiness,
            "score": self.game.current_scores,
            "total_lines": self.game.current_lines,
            # "row_transitions": row_transition,
            # "col_transitions": col_transition,
            # "cumulative_wells": cumulated_wells,
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
        # 0 no rotation + noop
        # 1 no rotation + right
        # 2 no rotation + left
        # 3 rotate 90 + noop
        # 4 rotate 90 + right
        # 5 rotate 90 + left
        # 6 rotate 180 + noop
        # 7 rotate 180 + right
        # 8 rotate 180 + left
        # 9 rotate -90 + noop
        # 10 rotate -90 + right
        # 11 rotate -90 + left
        # 12 hard drop
        
        if action == 0:
            pass

        if action in [1, 2]:
            self.game.input(1 if action == 1 else -1)

        if action in [3, 9]:
            self.game.tetromino.rotate("right" if action == 3 else "left")

        if action in [4, 5, 10, 11]:
            self.game.tetromino.rotate("right" if action <= 5 else "left")
            self.game.input(1 if action in [4, 10] else -1)

        if action == 6:
            self.game.tetromino.rotate("right", amount=2)

        if action in [7, 8]:
            self.game.tetromino.rotate("right", amount=2)
            self.game.input(1 if action == 7 else -1)

        if action == 12:  # drop
            self.game.drop()

        self.render()

        info = self._get_info()
        observation = self._get_obs()

        reward = self.evaluate(info)

        return observation, reward, self.game.tetromino.game_over, False, info

    def evaluate(self, info):
        # reward = (
        #     -4 * info["holes"]
        #     - info["cumulative_wells"]
        #     - info["row_transitions"]
        #     - info["col_transitions"]
        #     - self.game.last_landing_height
        # )

        reward = - 0.51 * info["heights"] + 0.76 * info["total_lines"] - 0.36 * info["holes"] - 0.18 * info["bumpiness"]

        if self.game.tetromino.game_over:
            return -2
        else:
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

            canvas.fill((67, 70, 75))
            sfc = self.game.run(canvas)
            self.score.run(canvas)
            self.preview.run(self.next_shapes, canvas)

            matrix_screen = pygame.transform.rotate(canvas.subsurface(
                pygame.Rect(PIXEL, PIXEL, MATRIX_WIDTH, MATRIX_HEIGHT)
            ), 90)

            self.matrix_screen_array = pygame.surfarray.array3d(matrix_screen)

            self.game.draw_line(canvas, sfc)

            if self.game.tetromino.game_over:
                if self.game_over_screen is None:
                    self.game_over_screen = os.path.join(IMG_DIR, "game_over.jpg")
                    pygame.image.save(canvas, self.game_over_screen)
                    self.game_over_screen = pygame.image.load(self.game_over_screen)
                font = pygame.font.Font(None, 32)  # Adjust font size as needed
                text_surface = font.render("Game Over", True, (255, 255, 255))  # Red color

                # Get text surface dimensions and screen center
                text_width, text_height = text_surface.get_size()
                screen_width, screen_height = canvas.get_size()
                center_x = (screen_width - text_width) // 2
                center_y = (screen_height - text_height) // 2

                canvas.blit(self.game_over_screen, (0, 0))
                canvas.blit(text_surface, (center_x, center_y))

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
