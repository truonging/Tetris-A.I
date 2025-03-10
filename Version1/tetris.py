from settings import *
from sys import exit
import os

from game import Game
from scoreboard import Scoreboard
from preview import Preview
from ai_board import AI_board


class Tetris:
    def __init__(self,i=1,SLOW_DROP=True):
        self.SLOW_DROP = SLOW_DROP
        self.set_window_position(i)
        pygame.init()
        if not RENDER:
            pygame.display.init()
        else:
            self.display_surface = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Tetris')
        self.games = 1

        self.game = Game()
        if RENDER:
            self.scoreboard = Scoreboard()
            #self.preview = Preview()
            #self.ai_board = AI_board()
        self.reward = 0
        self.confidence = 0
        self.random = False
        self.state = self.game.get_state()
        self.hard_drop_on = True
        self.epsilon = 0

        if not RENDER:
            pygame.display.quit()

    def set_window_position(self, index):
        # Calculate grid positions for a 2x5 grid
        # index = 0
        tile_width, tile_height = GAME_WIDTH+SIDEBAR_WIDTH+(PADDING*2), GAME_HEIGHT+(PADDING*4)
        gap_offset = SIDEBAR_WIDTH
        rows, cols = 2, 5
        index %= 10
        row = index // cols
        col = index % cols
        x = col * tile_width #+ (5*tile_width+gap_offset)
        y = row * tile_height + 25
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"

    def reset(self):
        self.game = Game()
        if RENDER:
            # self.scoreboard = Scoreboard()
            self.scoreboard.reset()
            #self.preview = Preview()
            #self.preview.run(self.game)
            #self.ai_board = AI_board()
            #self.ai_board.reset()
        self.reward = self.confidence = 0
        self.random = False

    def update_rewards(self, reward):
        self.reward = reward

    def update_state(self, state, confidence,randomm, epsilon):
        self.state = state
        self.confidence = confidence
        self.random = randomm
        self.epsilon = epsilon

    def update_game_speed(self):
        # slow_games = {
        #     1, 2, 3, 4, 5, 25, 50, 100, 350, 500, 750,
        #     1001, 1002, 1003,
        # }
        slow_games = set()
        slow_score = 25_000_000

        if self.games in slow_games or self.SLOW_DROP:# or self.game.score >= slow_score:
            self.hard_drop_on = False# if self.game.score <= 30_000_000 else True
            self.clock.tick(40)
        else:
            self.hard_drop_on = True #if (self.games>50) else False
            self.clock.tick(0)

    def play_full(self, best_action):
        """
        default = 0
        left = -1
        right = 1
        down = 2
        rotate = 3
        """
        x,rotations = best_action
        reward = 0
        self.game.reset_turn_info()

        if self.game.board[0][4] or self.game.board[0][5] or self.game.board[0][6]:
            self.game.set_gameover(True)
            if RENDER:
                pygame.display.update()
            return reward, self.game.finished

        # rotate
        self.game.dirs = 3
        for rotate in range(int(rotations)):
            self.play_step()
        if self.random:
            self.game.tetromino.set_outline()

        # go towards the x-axis
        if not self.hard_drop_on:
            steps_needed = x - min(block.pos.x for block in self.game.tetromino.blocks)
            self.game.dirs = -1 if steps_needed < 0 else 1
            for step in range(abs(int(steps_needed))):
                self.play_step()
        else:
            if not self.game.tetromino.move_to_x(x):
                self.game.set_gameover(True)
                if RENDER:
                    pygame.display.update()
                return reward, self.game.finished

        # continue dropping until collision
        if not self.hard_drop_on:
            self.game.dirs = 2
            while not self.game.collision:
                self.play_step()
        else:
            self.game.dirs = 4
            self.play_step()

        self.game.dirs = 0

        # end of move
        return reward, self.game.finished

    def play_step(self):
        if RENDER:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    pygame.quit()
                    exit()
            self.display_surface.fill(GRAY)

        self.game.run()

        if RENDER:
            self.scoreboard.run(self.game, self.games, self.epsilon)
            pygame.display.update()

        self.update_game_speed()


if __name__ == '__main__':
    pass
    # tetris = Tetris()
    # tetris.play_full((0,1))
    #tetris.game.get_state()
    # tetris.play_full((2, 1))
    # tetris.play_full((4, 1))
    # tetris.play_full((6, 1))
    # tetris.play_full((7, 1))
    # print(tetris.game.get_state())
    # for i in range(4):
    #     tetris.play_full((i*2, 0))
    # for i in range(5):
    #     tetris.play_full((1+(i*2), 2))
    # for i in range(3):
    #     tetris.play_full((1+i*3, 3))
    # for i in range(3):
    #     tetris.play_full((i*3,0))
    #
    # tetris.play_full((9,2))
    # tetris.play_full((4, 0))