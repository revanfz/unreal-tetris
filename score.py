import pygame
from settings import *

class Score:
    def __init__(self):
        self.surface = pygame.Surface((SIDEBAR_WIDTH, MATRIX_HEIGHT * SCOREBAR_HEIGHT - PIXEL))
        self.display_surface = pygame.display.get_surface()
        self.rect = self.surface.get_rect(bottomright=(WINDOW_WIDTH - PIXEL, WINDOW_HEIGHT - PIXEL))

    def run(self):
        self.display_surface.blit(self.surface, self.rect)