from numpy import block
import pygame
from settings import *


class Tetromino:
    def __init__(self, shape, group, create_new_tetromino):
        self.block_position = TETROMINOS[shape]['shape']
        self.color = TETROMINOS[shape]['color']
        self.blocks = [Block(group, pos, self.color) for pos in self.block_position]
        self.create_new_tetromino = create_new_tetromino


    def move_down(self):
        if self.check_vertical_collision(1):
            print(f'Vertical Collision')
            self.create_new_tetromino()
        else:
            for block in self.blocks:
                block.pos.y += 1

    def move_horizontal(self, amount):
        if self.check_horizontal_collision(amount):
            print(f'Horizontal Collision')
        else:
            for block in self.blocks:
                block.pos.x += amount

    def check_horizontal_collision(self, amount):
        collision_list = [block.horizontal_collide(int(block.pos.x + amount)) for block in self.blocks]
        return True if sum(collision_list) else False

    def check_vertical_collision(self, amount):
        collision_list = [block.vertical_collide(int(block.pos.y + amount)) for block in self.blocks]
        return True if sum(collision_list) else False


class Block(pygame.sprite.Sprite):
    def __init__(self, group, pos, color):
        super().__init__(group)
        self.image = pygame.Surface((PIXEL, PIXEL))
        self.image.fill(color)
        self.pos = pygame.Vector2(pos) + pygame.Vector2(COL // 2 - 1, -1)

        self.rect = self.image.get_rect(topleft = self.pos * PIXEL)

    def update(self):
        self.rect.topleft = self.pos * PIXEL

    def horizontal_collide(self, target: int):
        return False if 0 <= target < COL else True
    
    def vertical_collide(self, target: int):
        return True if target >= ROW else False