import pygame
from .settings import *


class Tetromino:
    def __init__(self, shape, group, create_new_tetromino, field_data):
        self.shape = shape
        self.block_type = TETROMINOS[shape]["id"]
        self.block_position = TETROMINOS[shape]["shape"]
        self.color = TETROMINOS[shape]["color"]
        self.image = BLOCK_IMG_DIR + "/" + TETROMINOS[shape]["image"]
        self.blocks = [
            Block(group, pos, self.color, self.image) for pos in self.block_position
        ]
        self.create_new_tetromino = create_new_tetromino
        self.field_data = field_data
        self.game_over = False

    def drop_shape(self):
        # y baris, x kolom
        lowest_row_dict = {}

        for block in self.blocks:
            # ekstrak nilai baris tertinggi yang berisi pada setiap kolom
            if (
                block.pos.x not in lowest_row_dict
                or block.pos.y > lowest_row_dict[block.pos.x]
            ):
                lowest_row_dict[block.pos.x] = block.pos.y

        # mapping jarak dari baris yang sudah terisi dengan shape yang akan turun
        distance_dict = {}
        for item in set(block.pos.x for block in self.blocks):
            temp = next(
                (i for i, row in enumerate(self.field_data) if row[int(item)] != 0),
                20,
            )
            if temp is not None:
                distance_dict[item] = temp - lowest_row_dict[item]

        # mengambil jarak yang terdekat untuk collision
        distance = min(distance_dict.values())

        for block in self.blocks:
            block.pos.y += distance - 1

        self.move_down()

    def set_game_over(self):
        self.game_over = True

    def move_down(self):
        if self.check_vertical_collision(1):
            for block in self.blocks:
                self.field_data[int(block.pos.y)][int(block.pos.x)] = block

            if self.check_ceiling_collision():
                self.set_game_over()
            else:
                self.create_new_tetromino()
        else:
            for block in self.blocks:
                block.pos.y += 1

    def move_horizontal(self, amount):
        if self.check_horizontal_collision(amount):
            return True
        else:
            for block in self.blocks:
                block.pos.x += amount

        return False

    def check_horizontal_collision(self, amount):
        collision_list = [
            block.horizontal_collide(int(block.pos.x + amount), self.field_data)
            for block in self.blocks
        ]
        return True if sum(collision_list) else False

    def check_vertical_collision(self, amount):
        collision_list = [
            block.vertical_collide(int(block.pos.y + amount), self.field_data)
            for block in self.blocks
        ]
        return True if sum(collision_list) else False

    def check_ceiling_collision(self):
        for blocks in self.blocks:
            if blocks.pos.y <= 0:
                return True

        return False

    def rotate(self, direction, amount=1):
        if self.shape != "O":
            pivot_pos = self.blocks[0].pos  # pivot point
            # new block position after rotating
            new_block_position = [
                block.rotate(pivot_pos, direction, amount) for block in self.blocks
            ]

            # check collision
            for pos in new_block_position:
                # horizontal
                if pos.x < 0:
                    distance = 0 - pos.x
                    for position in new_block_position:
                        position.x += distance

                elif pos.x >= COL:
                    distance = pos.x - COL + 1
                    for position in new_block_position:
                        position.x -= distance

                # vertical / floor
                if pos.y >= ROW:
                    distance = pos.y - ROW + 1
                    for position in new_block_position:
                        position.y -= distance

            for pos in new_block_position:
                # field check (with other pieces)
                if self.field_data[int(pos.y)][int(pos.x)]:
                    return True

            for i, block in enumerate(self.blocks):
                block.pos = new_block_position[i]

        return False


class Block(pygame.sprite.Sprite):
    def __init__(self, group, pos, color, image):
        super().__init__(group)
        self.color = color
        self.image = image
        image = pygame.image.load(image)
        image = pygame.transform.scale(image, (PIXEL, PIXEL))
        surface = pygame.Surface([PIXEL, PIXEL])
        # surface.fill(color=self.color)
        surface.blit(image, (0 ,0))
        self.pos = pygame.Vector2(pos) + pygame.Vector2(COL // 2 - 1, -1)

        self.rect = surface.get_rect(topleft=self.pos * PIXEL)

    def rotate(self, pivot_pos, direction, amount=1):
        multiplication = -1 if direction == "left" else 1
        return pivot_pos + (self.pos - pivot_pos).rotate(90 * multiplication * amount)

    def update(self):
        self.rect.topleft = self.pos * PIXEL

    def horizontal_collide(self, target: int, field_data):
        if not 0 <= target < COL:
            return True
        if field_data[int(self.pos.y) if self.pos.y > 0 else 0][target]:
            return True

        return False

    def vertical_collide(self, target: int, field_data):
        if target >= ROW:
            return True

        if target >= 0 and field_data[target][int(self.pos.x)]:
            return True

        return False
