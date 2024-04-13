import pygame

from os import path
from .settings import *
from pygame.image import load
from pygame.transform import scale_by


class Preview:
    def __init__(self):
        self.surface = pygame.Surface(
            (SIDEBAR_WIDTH, PREVIEW_HEIGHT * WINDOW_HEIGHT - PIXEL)
        )
        self.rect = self.surface.get_rect(topright=(WINDOW_WIDTH - PIXEL, PIXEL))
        self.display_surface = pygame.display.get_surface()

        self.shape_surfaces = {
            shape: scale_by(
                load(path.join(TETROMINOS_IMG_DIR, f"{shape}.png")),
                0.35,
            ).convert_alpha()
            for shape in TETROMINOS.keys()
        }

        self.fragment_height = self.surface.get_height() / 3

    def display_pieces(self, next_shapes):
        for i, shape in enumerate(next_shapes):
            shape_surface = self.shape_surfaces[shape]
            x = self.surface.get_width() / 2
            y = self.fragment_height / 2 + i * self.fragment_height
            rect = shape_surface.get_rect(center=(x, y))
            self.surface.blit(shape_surface, rect)

    def run(self, next_shapes):
        self.surface.fill((67, 70, 75))
        self.display_pieces(next_shapes)
        self.display_surface.blit(self.surface, self.rect)
        pygame.draw.rect(self.display_surface, "WHITE", self.rect, 2, 2)
