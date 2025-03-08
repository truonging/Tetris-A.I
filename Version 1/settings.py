import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame


RENDER = True
HARD_DROP_ON = True
FRAMERATES = 0

# Game Size
COLUMNS = 10
ROWS = 20
CELL_SIZE = 30
SCALED_SIZE = CELL_SIZE / 40
GAME_WIDTH, GAME_HEIGHT, = COLUMNS*CELL_SIZE, ROWS*CELL_SIZE

# Sidebar size
SIDEBAR_WIDTH = CELL_SIZE*10
PREVIEW_HEIGHT_FRACTION = 0.4
SCORE_HEIGHT_FRACTION = 0.4
AI_HEIGHT_FRACTION = 0.2

# Window
PADDING = CELL_SIZE//2
WINDOW_WIDTH = GAME_WIDTH + SIDEBAR_WIDTH + PADDING*3
WINDOW_HEIGHT = GAME_HEIGHT + PADDING*2

# game behavior
UPDATE_START_SPEED = 200
MOVE_WAIT_TIME = 200
ROTATE_WAIT_TIME = 100
HARD_DROP_WAIT_TIME = 400
BLOCK_OFFSET = pygame.Vector2(4, -1)

# Colors
YELLOW = '#f1e60d'
PURPLE = '#7b217f'
RED = '#e51b20'
BLUE = '#204b9b'
GREEN = '#65b32e'
GRAY = '#1C1C1C'
ORANGE = '#f07e13'
CYAN = '#6cc6d9'
LINE_COLOR = '#FFFFFF'

# shapes
TETROMINOS = {
    'T': {'shape': [(0,0), (-1,0),(1,0),(0,-1)], 'color':PURPLE, 'rotations':4},
    'O': {'shape': [(0,0), (0,-1),(1,0),(1,-1)], 'color':YELLOW, 'rotations':1},
    'J': {'shape': [(0,0), (0,-1),(0,1),(-1,1)], 'color':BLUE, 'rotations':4},
    'L': {'shape': [(0,0), (0,-1),(0,1),(1,1)], 'color':ORANGE, 'rotations':4},
    'I': {'shape': [(0,0), (0,-1),(0,-2),(0,1)], 'color':CYAN, 'rotations':2},
    'S': {'shape': [(0,0), (-1,0),(0,-1),(1,-1)], 'color':GREEN, 'rotations':2},
    'Z': {'shape': [(0,0), (1,0),(0,-1),(-1,-1)], 'color':RED, 'rotations':2},
}
TETROMINOS_KEYS = ['T','O','J','L','I','S','Z']
KEY_LEN = len(TETROMINOS_KEYS)

SCORE_DATA = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}

DIRS = [0,1,0,-1,0]
BLOCK_DIRS = [("left", (-1, 0)),("right", (1, 0)),("up", (0, -1)),("down", (0, 1))]

calc_pos = lambda block, piv: piv + (block.pos - piv).rotate(90)