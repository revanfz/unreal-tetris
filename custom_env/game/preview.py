import pygame

from .settings import *
from pygame.image import load
from pygame.transform import scale_by


class Preview:
    def __init__(self):
        pass

    def display_pieces(self, next_shapes, surface, portion, shape_surfaces):
        for i, shape in enumerate(next_shapes):
            shape_surface = shape_surfaces[shape]
            x = surface.get_width() / 2
            y = portion / 2 + i * portion
            rect = shape_surface.get_rect(center=(x, y))
            surface.blit(shape_surface, rect)

    def run(self, next_shapes, display_surface):
        shape_surfaces = {
            shape: scale_by(
                pygame.image.load(TETROMINOS_IMG_DIR + f"/{shape}.png"), 0.1
            )
            for shape in TETROMINOS.keys()
        }
        surface = pygame.Surface(
            (SIDEBAR_WIDTH, PREVIEW_HEIGHT * WINDOW_HEIGHT - PIXEL)
        )
        rect = surface.get_rect(topright=(WINDOW_WIDTH - PIXEL, PIXEL))
        fragment_height = surface.get_height() / 3
        surface.fill((67, 70, 75))
        self.display_pieces(next_shapes, surface, fragment_height, shape_surfaces)
        display_surface.blit(surface, rect)
        pygame.draw.rect(display_surface, "WHITE", rect, 2, 2)
