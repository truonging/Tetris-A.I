import numpy as np
import pygame

from settings import *
from random import random

class Tetris:
    def __init__(self, i=1):
        self.board = [[0]*COLS for _ in range(ROWS)]
        self.prev_board = [row[:] for row in self.board]

        # display
        self.i = i

        self.tetromino = self.blocks = None
        self.create_tetromino()

        self.total_lines = 0
        self.lines = 0
        self.lines_removed = 0
        self.finished = False
        self.collision = False
        self.total_games = 1
        self.old_state = [0, 0, 0, 0, 0,0]
        self.score = 0
        self.hiscore = 0

    def reset(self):
        self.board = [[0]*COLS for _ in range(ROWS)]
        self.create_tetromino()
        self.lines = 0
        self.lines_removed = 0
        self.finished = False
        self.collision = False
        self.score = 0

    def clear_board(self):
        changed_blocks = []
        for r in range(ROWS):
            for c in range(COLS):
                changed_blocks.append((self.i,r,c,0))
        self.prev_board = [[0]*COLS for _ in range(ROWS)]
        return changed_blocks

    def create_tetromino(self):
        self.tetromino = TETROMINOS[TETROMINOS_KEYS[int(random()*KEY_LEN)]]
        self.blocks = self.tetromino['shape'][0][:]
        if any(self.blocks[i*2]>=0 and self.board[self.blocks[i*2]][self.blocks[i*2+1]] for i in range(4)):
            self.finished = True
            return False
        return True

    def delete_lines(self, board):
        delete_rows = set()
        for i in range(4):
            if self.blocks[i*2] not in delete_rows and all(board[self.blocks[i*2]]):
                delete_rows.add(self.blocks[i*2])
        if delete_rows:
            board = [[0]*COLS for _ in range(len(delete_rows))] + [board[r] for r in range(ROWS) if r not in delete_rows]
        return board, len(delete_rows)

    def update_board(self, board=None, update_game=True):
        if not board:
            board = self.board
        # place the falling piece on board
        for i in range(4):
            board[self.blocks[i*2]][self.blocks[i*2+1]] = self.tetromino['color_id']
        board, lines_removed = self.delete_lines(board)

        if update_game:
            self.board = board
            self.lines_removed = lines_removed
            self.lines += lines_removed
            self.total_lines += lines_removed
            self.score += SCORE_DATA[lines_removed] * (self.lines // 10 + 1)
            self.hiscore = max(self.hiscore, self.score)

        return board, lines_removed

    def hard_drop(self, board=None, update_game=True):
        while True:
            if self.move_down(board, update_game):
                return

    def move_down(self, board=None, update_game=True):
        if not board:
            board = self.board
        # check if u can move down one
        if any(self.blocks[i*2]+1>=ROWS or (self.blocks[i*2]+1>=0 and board[self.blocks[i*2]+1][self.blocks[i*2+1]]) for i in range(4)):
            if update_game:
                self.collision = True
                if any(self.blocks[i*2]<0 for i in range(4)):
                    self.finished = True
            return True

        for i in range(4):
            self.blocks[i*2] += 1

        return False

    def move_to_x(self, x):
        diff = x - min(self.blocks[1::2])
        block = self.blocks
        if any((not 0<=block[i*2+1]+diff<=9) or (block[i*2]>=0 and self.board[block[i*2]][block[i*2+1]+diff]>0) for i in range(4)):
            return False
        for i in range(4):
            self.blocks[i*2+1] += diff
        return True

    def move_to_y(self, y):
        diff = self.blocks[0] - y
        for i in range(4):
            self.blocks[i*2] -= diff

    def play(self, action):
        x, rotations = action
        rotations *= 90

        self.collision = False
        self.lines_removed = 0

        self.blocks = self.tetromino['shape'][rotations][:]

        self.move_to_x(x)

        while not self.collision:
            self.move_down()

        self.update_board()

        changes = self.add_changes()

        self.create_tetromino()

        return changes

    def add_changes(self):
        """
        if not cleared, just add the piece
        else, loop through board to determine which changed
        anything that changed, change to the current board[r][c]
        """
        changed_blocks = []
        board = self.board
        prev = self.prev_board
        if self.lines_removed:
            for r in range(ROWS):
                for c in range(COLS):
                    if board[r][c]!=prev[r][c]:
                        changed_blocks.append((self.i,r,c,board[r][c]))
                        prev[r][c] = board[r][c]
        else:
            for i in range(4):
                r,c = self.blocks[i*2],self.blocks[i*2+1]
                color = self.tetromino['color_id']
                changed_blocks.append((self.i,r,c,color))
                prev[r][c] = color
        return changed_blocks


import cProfile
import pstats
if __name__=='__main__':

    tetris = Tetris(1)
    tetris.tetromino = TETROMINOS['S']
    tetris.board = [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0, 1, 0, 0, 1, 1],
        [0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    ]
    rotate_shapes = TETROMINOS_NUMPY[tetris.tetromino['key']]['shape']
    # next_states = {tuple(v): k for k, v in game.calc_all_states().items()}
    # a = tetris.calc_all_states()
    # # # b = tetris.calc_all_states2()
    c = calc_all_states3(
        np.array(tetris.board, dtype=np.int64),
        tetris.tetromino['rotations'],
        rotate_shapes[0],
        rotate_shapes[90],
        rotate_shapes[180],
        rotate_shapes[270]
    )
    # # # print(a==b)
    # print(a==c)
    # # # print(b==c)
    # cProfile.run('run_game()', 'profile_output.prof')
    # p = pstats.Stats('profile_output.prof')
    # p.strip_dirs().sort_stats('cumulative').print_stats(lambda x: x >= 1)