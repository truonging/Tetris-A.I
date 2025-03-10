from settings import *
from sys import exit

from game import Game
from scoreboard import Scoreboard
from preview import Preview
from ai_board import AI_board

class Tetris:
    def __init__(self):
        pygame.init()
        self.display_surface = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Tetris')

        self.games = 1
        self.game = Game(player=True)
        self.scoreboard = Scoreboard()
        self.preview = Preview()
        self.ai_board = AI_board()
        self.reward = 0
        self.confidence = 0
        self.random = False
        self.state = self.game.get_state()

    def reset(self):
        self.game = Game(player=True)
        self.scoreboard = Scoreboard()
        self.preview = Preview()
        self.ai_board = AI_board()
        self.reward = self.confidence = 0
        self.random = False

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    pygame.quit()
                    exit()

            # display
            self.display_surface.fill(GRAY)

            self.game.run()
            if self.game.finished:
                self.reset()

            self.scoreboard.run(self.game, self.games)
            self.preview.run(self.game)
            self.ai_board.run(self.game,self.reward,self.confidence,self.state,self.random)

            # updating the game
            pygame.display.update()
            self.clock.tick()

if __name__ == '__main__':
    tetris = Tetris()
    tetris.run()