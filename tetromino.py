import pygame
from settings import *


class Tetromino:
    def __init__(self, shape, group, create_new_tetromino, field_data):
        self.block_position = TETROMINOS[shape]['shape']
        self.color = TETROMINOS[shape]['color']
        self.blocks = [Block(group, pos, self.color) for pos in self.block_position]
        self.create_new_tetromino = create_new_tetromino
        self.field_data = field_data


    def move_down(self):
        if self.check_vertical_collision(1):
            print(f'Vertical Collision')
            for block in self.blocks:
                self.field_data[int(block.pos.y)][int(block.pos.x)] = block
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
        collision_list = [block.horizontal_collide(int(block.pos.x + amount), self.field_data) for block in self.blocks]
        return True if sum(collision_list) else False

    def check_vertical_collision(self, amount):
        collision_list = [block.vertical_collide(int(block.pos.y + amount), self.field_data) for block in self.blocks]
        return True if sum(collision_list) else False


class Block(pygame.sprite.Sprite):
    def __init__(self, group, pos, color):
        super().__init__(group)
        self.image = pygame.Surface((PIXEL, PIXEL))
        self.image.fill(color)
        # self.image = pygame.image.load("./shape.png")
        self.pos = pygame.Vector2(pos) + pygame.Vector2(COL // 2 - 1, -1)

        self.rect = self.image.get_rect(topleft = self.pos * PIXEL)

    def update(self):
        self.rect.topleft = self.pos * PIXEL

    def horizontal_collide(self, target: int, field_data):
        if not 0 <= target < COL:
            return True
        if field_data[int(self.pos.y)][target]:
            return True
        
        return False
    
    def vertical_collide(self, target: int, field_data):
        if target >= ROW:
            return True
        
        if target >= 0 and field_data[target][int(self.pos.x)]:
            return True
        
        return False