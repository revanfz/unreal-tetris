import pygame
from sys import exit
from preview import Preview
from score import Score
from settings import *
from matrix import Matrix


class Main:
    def __init__(self):
        pygame.init()
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Tetris Smart Agent')
        
        self.game = Matrix()
        self.score = Score()
        self.preview = Preview()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        print("Up Button")
                    if event.key == pygame.K_RIGHT:
                        self.game.input(1)
                    if event.key == pygame.K_DOWN:
                        print("Down Button")
                    if event.key == pygame.K_LEFT:
                        self.game.input(-1)

            self.display_surface.fill((67, 70, 75))
            self.game.run()
            self.score.run()
            self.preview.run()
            pygame.display.update()
            self.clock.tick(FPS)
    
if __name__ == '__main__':
    main = Main()
    main.run()