import pygame

from .settings import *
from pygame.image import load
from pygame.transform import scale_by


class Preview:
    def __init__(self):
        pass

    def display_pieces(self, next_shapes, current_pieces, surface, portion, shape_surfaces):
        shapes = [current_pieces] + next_shapes[:2]
        for i, shape in enumerate(shapes):
            shape_surface = shape_surfaces[shape]
            x = surface.get_width() / 2
            y = portion / 2 + i * portion
            rect = shape_surface.get_rect(center=(x, y))
            surface.blit(shape_surface, rect)

    def run(self, next_shapes, current_pieces, display_surface):
        shape_surfaces = {
            shape: scale_by(
                pygame.image.load(TETROMINOS_IMG_DIR + f"/{shape}.png"), 0.18
            )
            for shape in TETROMINOS.keys()
        }
        surface = pygame.Surface(
            (SIDEBAR_WIDTH, PREVIEW_HEIGHT * WINDOW_HEIGHT - PIXEL)
        )
        rect = surface.get_rect(topright=(WINDOW_WIDTH - PIXEL, PIXEL))
        fragment_height = surface.get_height() / 3

        red_surface = pygame.Surface((surface.get_width(), fragment_height))
        red_surface.fill("#2EB774")  # Warna merah
        
        # Buat permukaan dengan latar belakang abu-abu untuk 2/3 bagian sisanya
        gray_surface = pygame.Surface((surface.get_width(), 2 * fragment_height))
        gray_surface.fill((67, 70, 75))  # Warna abu-abu
        
        # Tempatkan permukaan-permukaan ini pada permukaan utama
        surface.blit(red_surface, (0, 0))  # Tempatkan permukaan merah di bagian atas
        surface.blit(gray_surface, (0, fragment_height))  # Tempatkan permukaan abu-abu di bagian bawah
    
        # surface.fill((67, 70, 75))
        self.display_pieces(next_shapes, current_pieces, surface, fragment_height, shape_surfaces)
        display_surface.blit(surface, rect)
        pygame.draw.rect(display_surface, "WHITE", rect, 2, 2)
