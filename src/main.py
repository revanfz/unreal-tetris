import pygame
from sys import exit
from preview import Preview
from score import Score
from settings import *
from matrix import Matrix
from random import choice


class Main:
    def __init__(self):
        pygame.init()
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Tetris Smart Agent')

        self.next_shapes = [choice(list(TETROMINOS.keys())) for shape in range(3)]
        # print(self.next_shapes)
        
        self.game = Matrix(self.get_next_shape, self.update_score)
        self.score = Score()
        self.preview = Preview()

    def update_score(self, lines, scores, level):
        self.score.lines = lines
        self.score.level = level
        self.score.scores = scores

    def get_next_shape(self):
        next_shape = self.next_shapes.pop(0)
        self.next_shapes.append(choice(list(TETROMINOS.keys())))
        return next_shape

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        if not self.game.timers['rotate'].active:
                            self.game.tetromino.rotate()
                            self.game.timers['rotate'].activate()
                
                    if event.key == pygame.K_DOWN:
                        if not self.game.speedup:
                            self.game.speedup = True
                            self.game.timers['verticalMove'].duration = self.game.down_speed_faster
                
                    if not self.game.timers['horizontalMove'].active:
                        if event.key == pygame.K_RIGHT:
                            self.game.input(1)
                            self.game.timers['horizontalMove'].activate()
                    
                        if event.key == pygame.K_LEFT:
                            self.game.input(-1)
                            self.game.timers['horizontalMove'].activate()
                
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_DOWN:
                        if self.game.speedup:
                            self.game.speedup = False
                            self.game.timers['verticalMove'].duration = self.game.down_speed


            self.display_surface.fill((67, 70, 75))
            self.game.run()
            self.score.run()
            self.preview.run(self.next_shapes)
            pygame.display.update()
            self.clock.tick(FPS)
    
if __name__ == '__main__':
    main = Main()
    main.run()