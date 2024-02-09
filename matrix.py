from random import choice
import pygame
from settings import *
from tetromino import Tetromino
from timer import Timer


class Matrix:
    def __init__(self):
        self.surface = pygame.Surface((MATRIX_WIDTH, MATRIX_HEIGHT))
        self.display_surface = pygame.display.get_surface()
        self.rect = self.surface.get_rect(topleft = (PIXEL, PIXEL))
        self.sprites = pygame.sprite.Group()

        self.line_surface = self.surface.copy()
        self.line_surface.fill((0, 255, 0))
        self.line_surface.set_colorkey((0, 255, 0))
        self.line_surface.set_alpha(125)

        self.create_new_tetromino()

        self.timers = {
            'vertical_move': Timer(UPDATE_START_SPEED, True, self.move_down)
        }
        self.timers['vertical_move'].activate()

    def create_new_tetromino(self):
        self.tetromino = Tetromino(choice(list(TETROMINOS.keys())), self.sprites, self.create_new_tetromino)

    def timer_update(self):
        for timer in self.timers.values():
            timer.update()

    def input(self, amount):
        self.tetromino.move_horizontal(amount)

    def move_down(self):
        self.tetromino.move_down()


    def draw_pixel(self):
        for col in range(1, COL):
            x = col * PIXEL
            pygame.draw.line(self.line_surface, 'WHITE', (x, 0), (x, self.surface.get_height()), 1) 

        for row in range(1, ROW):
            y = row * PIXEL
            pygame.draw.line(self.line_surface, 'WHITE', (0, y), (self.surface.get_width(), y))    

        self.surface.blit(self.line_surface, (0, 0))


    def run(self):
        self.timer_update()
        self.sprites.update()
        self.surface.fill((67, 70, 75))
        self.sprites.draw(self.surface)

        self.draw_pixel()
        self.display_surface.blit(self.surface, (PIXEL, PIXEL))
        pygame.draw.rect(self.display_surface, 'WHITE', self.rect, 2, 2)
