import pygame

from os.path import join
from .settings import *


class Score:
    def __init__(self):
        self.scores = 0
        self.level = 1
        self.lines = 0

    def run(self, display_surface):
        pygame.font.init()
        font = pygame.font.Font(join("custom_env", "game", "assets", "Tetris.ttf"), 8)

        surface = pygame.Surface(
            (SIDEBAR_WIDTH, MATRIX_HEIGHT * SCOREBAR_HEIGHT - PIXEL)
        )
        rect = surface.get_rect(
            bottomright=(WINDOW_WIDTH - PIXEL, WINDOW_HEIGHT - PIXEL)
        )
        fragment_height = surface.get_height() / 3
    

        surface.fill((67, 70, 75))
        for i, text in enumerate(
            [("Score", self.scores), ("Level", self.level), ("Lines", self.lines)]
        ):
            x = surface.get_width() / 2
            y = fragment_height / 2 + i * fragment_height
            text_surface = font.render(f"{text[0]}: {text[1]}", True, "white")
            text_rect = text_surface.get_rect(center=(x, y))
            surface.blit(text_surface, text_rect)

        display_surface.blit(surface, rect)
