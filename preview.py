import pygame
from settings import *

class Preview:
    def __init__(self):
        self.surface = pygame.Surface((SIDEBAR_WIDTH, PREVIEW_HEIGHT * WINDOW_HEIGHT - PIXEL))
        self.rect = self.surface.get_rect(topright=(WINDOW_WIDTH - PIXEL, PIXEL))
        self.display_surface = pygame.display.get_surface()

    def run(self):
        self.display_surface.blit(self.surface, self.rect)
