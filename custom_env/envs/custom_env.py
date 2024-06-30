import os
import sys
import torch
import random
import pygame
import numpy as np
import gymnasium as gym

sys.path.append("/")

from gymnasium.spaces import Box, Dict, Discrete

from custom_env.game.matrix import Matrix
from custom_env.game.settings import (
    MATRIX_WIDTH,
    MATRIX_HEIGHT,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    PIXEL,
    IMG_DIR,
)
from custom_env.game.preview import Preview
from custom_env.game.score import Score

from torchvision.transforms import v2


def transformImage(image):
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((84, 84)),
            v2.Grayscale(num_output_channels=1),
            v2.RandomHorizontalFlip(p=1)
        ]
    )

    return transform(image).numpy()


class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}

    def __init__(self, render_mode=None):
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("Tetris Smart Agent")

        self.window_size = (WINDOW_WIDTH, WINDOW_HEIGHT)

        self.observation_space = Dict(
            {
                "matrix_image": Box(
                    low=0,
                    high=1,
                    shape=(1, 84, 84),
                    dtype=np.float32,
                ),
                "falling_shape": Box(
                    low=0, high=1, shape=(1, 84, 84), dtype=np.float32
                ),
            }
        )
        self.action_space = Discrete(36)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.game_over_screen = None

    def get_next_shape(self):
        if (self.game.block_placed - 4) % 7 == 0:
            random.shuffle(self.bag)

        next_shape = self.next_shapes.pop(0)
        self.next_shapes.append(
            self.bag[
                (
                    (self.game.block_placed - 4) % 7
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

    def calculate_bump(self, col_heights):
        bumpiness = 0
        for i in range(len(col_heights) - 1):
            bumpiness += abs(col_heights[i] - col_heights[i + 1])

        return bumpiness

    def _get_info(self):
        col_heights = self.find_col_heights()
        holes = self.calculate_holes(col_heights)
        bumpiness = self.calculate_bump(col_heights)
        return {
            "heights": sum(col_heights),
            "lines_cleared": self.game.last_deleted_rows,
            "holes": sum(holes),
            "bumpiness": bumpiness,
            "score": self.game.current_scores,
            "total_lines": self.game.current_lines,
            "block_placed": self.game.block_placed,
            "falling_shape": self.game.tetromino.shape
        }

    def _get_obs(self):
        return {
            "matrix_image": transformImage(self.matrix_screen_array),
            "falling_shape": transformImage(self.falling_shape)
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
        """
        0 Noop             4 Rotasi kiri
        1 kanan            5 Soft drop
        2 Kiri             6 Hard drop
        3 Rotasi Kanan
        """

        if action == 0:
            pass

        elif action <= 35:
            if action % 4 != 0:
                self.game.tetromino.rotate("right" if action % 4 != 3 else "left", amount=2 if action % 4 == 2 else 1)
            
            step = (action // 4) # misal 22 -> 4, berarti 4 step ke kanan, kalo 32 -> berarti 8, 4 step ke kiri
            print(step)
            if step:
                step = step if step < 5 else step - 4
                for _ in range(step):
                    move = 1 if action < 20 else -1
                    self.game.input(move)
           
        self.game.drop()

        self.render()

        info = self._get_info()
        observation = self._get_obs()
        reward = self.evaluate(info)

        return observation, reward, self.game.tetromino.game_over, False, info

    def evaluate(self, info):
        # reward = 0
        reward = info["lines_cleared"] * 0.76 - (0.51 * info["heights"] + 0.36 * info["holes"] + 0.18 * info["bumpiness"])
        # if self.game.tetromino.game_over:
        #     return -1
        # else:
            # for i in range(info["lines_cleared"], 0, -1):
            #     reward += i

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
            self.preview.run(self.next_shapes, self.game.tetromino.shape, canvas)

            falling_shape = self.game.get_falling_block()

            self.falling_shape = pygame.surfarray.array3d(pygame.transform.rotate(falling_shape, 90))

            matrix_screen = pygame.transform.rotate(
                canvas.subsurface(
                    pygame.Rect(PIXEL, PIXEL, MATRIX_WIDTH, MATRIX_HEIGHT)
                ),
                90,
            )

            self.matrix_screen_array = pygame.surfarray.array3d(matrix_screen)
            self.game.draw_line(canvas, sfc)

            if self.game.tetromino.game_over:
                if self.game_over_screen is None and self.render_mode == "human":
                    self.game_over_screen = os.path.join(IMG_DIR, "game_over.jpg")
                    pygame.image.save(canvas, self.game_over_screen)
                    self.game_over_screen = pygame.image.load(self.game_over_screen)

                    canvas.blit(self.game_over_screen, (0, 0))

                font = pygame.font.Font(None, 32)  # Adjust font size as needed
                text_surface = font.render(
                    "Game Over", True, (255, 255, 255)
                )  # Red color

                # Get text surface dimensions and screen center
                text_width, text_height = text_surface.get_size()
                screen_width, screen_height = canvas.get_size()
                center_x = (screen_width - text_width) // 2
                center_y = (screen_height - text_height) // 2
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
