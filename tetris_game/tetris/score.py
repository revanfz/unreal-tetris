import pygame

from os.path import join
from .settings import *


class Score:
    def __init__(self):
        self.surface = pygame.Surface(
            (SIDEBAR_WIDTH, MATRIX_HEIGHT * SCOREBAR_HEIGHT - PIXEL)
        )
        self.display_surface = pygame.display.get_surface()
        self.rect = self.surface.get_rect(
            bottomright=(WINDOW_WIDTH - PIXEL, WINDOW_HEIGHT - PIXEL)
        )

        self.font = pygame.font.Font(join("tetris_game", "assets", "Tetris.ttf"), 16)

        self.fragment_height = self.surface.get_height() / 3

        self.scores = 0
        self.level = 1
        self.lines = 0

    def display_text(self, pos, text):
        text_surface = self.font.render(f"{text[0]}: {text[1]}", True, "white")
        text_rect = text_surface.get_rect(center=pos)
        self.surface.blit(text_surface, text_rect)

    def run(self):
        self.surface.fill((67, 70, 75))
        for i, text in enumerate(
            [("Score", self.scores), ("Level", self.level), ("Lines", self.lines)]
        ):
            x = self.surface.get_width() / 2
            y = self.fragment_height / 2 + i * self.fragment_height
            self.display_text((x, y), text)

        self.display_surface.blit(self.surface, self.rect)
