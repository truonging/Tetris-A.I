import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from random import random
from os.path import join
import math
import time
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import heapq
import matplotlib.pyplot as plt
from itertools import combinations

from numba import njit, types
import timeit

import numpy as np
from numba import njit
from numba.typed import Dict
from numba import types

RENDER = True
HARD_DROP = False
SCORE_BOARD_ON = False
FPS = 0

# Game Size
COLS = 10
ROWS = 20
CELL_SIZE = 4
GAME_WIDTH, GAME_HEIGHT, = COLS*CELL_SIZE, ROWS*CELL_SIZE

# Sidebar size
SIDEBAR_WIDTH = CELL_SIZE*10 if SCORE_BOARD_ON else 0
PREVIEW_HEIGHT_FRACTION = 0.3
SCORE_HEIGHT_FRACTION = 0.4
AI_HEIGHT_FRACTION = 0.3

# Window
PADDING = CELL_SIZE//2 #1
# WINDOW_WIDTH = GAME_WIDTH + SIDEBAR_WIDTH + PADDING*(3 if SCORE_BOARD_ON else 2)
# WINDOW_HEIGHT = GAME_HEIGHT + PADDING*2
WINDOW_WIDTH = GAME_WIDTH + SIDEBAR_WIDTH + PADDING*(3 if SCORE_BOARD_ON else 2)
WINDOW_HEIGHT = GAME_HEIGHT + PADDING*2

# Calculate grid size
MAIN_COLS = 5
MAIN_ROWS = 10
MAIN_SIDEBAR_WIDTH = 250
MAIN_WIDTH = WINDOW_WIDTH * MAIN_COLS + MAIN_SIDEBAR_WIDTH
MAIN_HEIGHT = WINDOW_HEIGHT * MAIN_ROWS

def calc_i(i):
    r = (i-1)//MAIN_COLS
    c = (i-1)%MAIN_COLS
    return c,r
# avg score for model 59 before removing epsilon
POPULATION_SIZE = 50
SIZE_PICK = 5
TOTAL_GAMES = 500
GAMES_RATE = 50
GENERATIONS = 100
GENERATION_RATE = 10

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
    'T': {
        'shape': {
            0: [0,5, 0,4, 0,6, -1,5],
            90: [0,5, -1,5, 1,5, 0,6],
            180: [0,5, 0,6, 0,4, 1,5],
            270: [0,5, 1,5, -1,5, 0,4]
        },
        'color':PURPLE,
        'color_id': 2,
        'rotations':4,
        'key':'T'
    },
    'O': {
        'shape': {
            0: [0,5, -1,5, 0,6, -1,6],
            90: [0,5, -1,5, 0,6, -1,6],
            180: [0,5, -1,5, 0,6, -1,6],
            270: [0,5, -1,5, 0,6, -1,6],
        },
        'color':YELLOW,
        'color_id': 3,
        'rotations':1,
        'key':'O'
    },
    'J': {
        'shape': {
            0: [0,5, -1,5, 1,5, 1,4],
            90: [0,5, 0,6, 0,4, -1,4],
            180: [0,5, 1,5, -1,5, -1,6],
            270: [0,5, 0,4, 0,6, 1,6]
        },
        'color':BLUE,
        'color_id': 4,
        'rotations':4,
        'key':'J'
          },
    'L': {
        'shape': {
            0: [0,5, -1,5, 1,5, 1,6],
            90: [0, 5, 0, 6, 0, 4, 1, 4],
            180: [0, 5, 1, 5, -1, 5, -1, 4],
            270: [0, 5, 0, 4, 0, 6, -1, 6]
        },
        'color':ORANGE,
        'color_id': 5,
        'rotations':4,
        'key':'L'
          },
    'I': {
        'shape': {
            0: [0,5, -1,5, -2,5, 1,5],
            90: [0, 5, 0, 4, 0, 3, 0, 6],
            180: [0, 5, -1, 5, -2, 5, 1, 5],
            270: [0, 5, 0, 4, 0, 3, 0, 6],
        },
        'color':CYAN,
        'color_id': 6,
        'rotations':2,
        'key':'I'
          },
    'S': {
        'shape': {
            0: [0,5, 0,4, -1,5, -1,6],
            90: [0,5, -1,5, 0,6, 1,6],
            180: [0,5, 0,4, -1,5, -1,6],
            270: [0,5, -1,5, 0,6, 1,6],
        },
        'color':GREEN,
        'color_id': 7,
        'rotations':2,
        'key':'S'
          },
    'Z': {
        'shape': {
            0: [0,5, 0,6, -1,5, -1,4],
            90: [0,5, 1,5, 0,6, -1,6],
            180: [0,5, 0,6, -1,5, -1,4],
            270: [0,5, 1,5, 0,6, -1,6],
        },
        'color':RED,
        'color_id': 8,
        'rotations':2,
        'key':'Z'
          },
}

TETROMINOS_NUMPY = {
    'T': {
        'shape': {
            0: np.array([0,5, 0,4, 0,6, -1,5], dtype=np.int64),
            90: np.array([0,5, -1,5, 1,5, 0,6], dtype=np.int64),
            180: np.array([0,5, 0,6, 0,4, 1,5], dtype=np.int64),
            270: np.array([0,5, 1,5, -1,5, 0,4], dtype=np.int64),
        },
        'color':PURPLE,
        'color_id': 2,
        'rotations':4
    },
    'O': {
        'shape': {
            0: np.array([0,5, -1,5, 0,6, -1,6], dtype=np.int64),
            90: np.array([0,5, -1,5, 0,6, -1,6], dtype=np.int64),
            180: np.array([0,5, -1,5, 0,6, -1,6], dtype=np.int64),
            270: np.array([0,5, -1,5, 0,6, -1,6], dtype=np.int64),
        },
        'color':YELLOW,
        'color_id': 3,
        'rotations':1
    },
    'J': {
        'shape': {
            0: np.array([-1,5, -2,5, 0,5, 0,4], dtype=np.int64),
            90: np.array([-1,5, -1,6, -1,4, -2,4], dtype=np.int64),
            180: np.array([-1,5, 0,5, -2,5, -2,6], dtype=np.int64),
            270: np.array([-1,5, -1,4, -1,6, 0,6], dtype=np.int64),
        },
        'color':BLUE,
        'color_id': 4,
        'rotations':4
          },
    'L': {
        'shape': {
            0: np.array([-1,5, -2,5, 0,5, 0,6], dtype=np.int64),
            90: np.array([-1, 5, -1, 6, -1, 4, 0, 4], dtype=np.int64),
            180: np.array([-1, 5, 0, 5, -2, 5, -2, 4], dtype=np.int64),
            270: np.array([-1, 5, -1, 4, -1, 6, -2, 6], dtype=np.int64),
        },
        'color':ORANGE,
        'color_id': 5,
        'rotations':4
          },
    'I': {
        'shape': {
            0: np.array([0,5, -1,5, -2,5, 1,5], dtype=np.int64),
            90: np.array([0, 5, 0, 4, 0, 3, 0, 6], dtype=np.int64),
            180: np.array([0, 5, -1, 5, -2, 5, 1, 5], dtype=np.int64),
            270: np.array([0, 5, 0, 4, 0, 3, 0, 6], dtype=np.int64),
        },
        'color':CYAN,
        'color_id': 6,
        'rotations':2
          },
    'S': {
        'shape': {
            0: np.array([0,5, 0,4, -1,5, -1,6], dtype=np.int64),
            90: np.array([0,5, -1,5, 0,6, 1,6], dtype=np.int64),
            180: np.array([0,5, 0,4, -1,5, -1,6], dtype=np.int64),
            270: np.array([0,5, -1,5, 0,6, 1,6], dtype=np.int64),
        },
        'color':GREEN,
        'color_id': 7,
        'rotations':2
          },
    'Z': {
        'shape': {
            0: np.array([0,5, 0,6, -1,5, -1,4], dtype=np.int64),
            90: np.array([0,5, 1,5, 0,6, -1,6], dtype=np.int64),
            180:np.array([0,5, 0,6, -1,5, -1,4], dtype=np.int64),
            270: np.array([0,5, 1,5, 0,6, -1,6], dtype=np.int64),
        },
        'color':RED,
        'color_id': 8,
        'rotations':2
          },
}
BLOCK_COLORS = {0:GRAY, 1:GRAY, 2:PURPLE, 3:YELLOW, 4:BLUE, 5:ORANGE, 6:CYAN, 7:GREEN, 8:RED}
TETROMINOS_KEYS = ['T','O','J','L','I','S','Z']
KEY_LEN = len(TETROMINOS_KEYS)

SCORE_DATA = {0: 0, 1: 40, 2: 100, 3: 300, 4: 1200}

RED = ['#FF0000',
'#CC0000',
'#990000',
'#660000',
'#330000']

@njit
def get_states(board, y_pos, lines_removed=0):

    cols = [0] * COLS
    holes = 0
    bumpiness = 0
    for c in range(len(board[0])):
        block = False
        for r in range(len(board)):
            if board[r][c ] >0 and not block:
                block = True
                cols[c] = 20 -r
            if not board[r][c] and block:
                holes += 1
        if c>0:
            bumpiness += abs(cols[c]-cols[c-1])

    total_heights = sum(cols)
    pillar = 0

    # Check pillars in the middle columns
    for i in range(1, len(cols) - 1):
        if (cols[i-1] - cols[i] >= 3) and (cols[i+1] - cols[i] >= 3):
            pillar = 1
            break

    # # Check edge cases for the first and last columns
    if pillar==0:
        if cols[1] - cols[0] >= 3 or cols[-2] - cols[-1] >= 3:
            pillar = 1

    y_pos = y_pos

    state = (total_heights, bumpiness, lines_removed, holes, y_pos, pillar)

    return state

board = np.array([[0,1]], dtype=np.int32)
get_states(board,0,0)

@njit
def fast_sample(batch_size: int, probabilities: np.ndarray) -> np.ndarray:
    cumulative_probs = np.cumsum(probabilities)
    random_vals = np.random.rand(batch_size)
    indices = np.searchsorted(cumulative_probs, random_vals)
    return indices

tst = [0.9574618339538574, 1.0780868530273438, 9.955280303955078, 4.444355487823486, 17.603099822998047, 23.681934356689453, 12.865041732788086, 7.406424522399902, 49.94914627075195, 18.603679656982422, 93.3079605102539, 25.95961570739746, 53.43779754638672, 49.54551696777344, 34.93963623046875, 35.934383392333984, 8.858987808227539, 88.19979858398438, 55.720149993896484, 18.617090225219727, 25.904464721679688, 99.78998565673828, 135.8281707763672, 137.27114868164062, 33.68687438964844, 73.42084503173828, 97.22067260742188, 100.05431365966797, 80.08049011230469, 80.46965026855469, 39.502105712890625, 76.57616424560547, 42.263248443603516, 32.66176986694336, 126.30216979980469, 123.22296142578125, 156.81532287597656, 113.5536880493164, 62.86906433105469, 41.545711517333984, 21.820024490356445, 62.11841583251953, 82.67048645019531, 115.14796447753906, 106.09163665771484, 145.66909790039062, 172.4192657470703, 178.6018829345703, 174.5615234375, 91.50312042236328, 40.710609436035156, 83.9059829711914, 92.8108139038086, 154.51458740234375, 120.5770034790039, 128.2279815673828, 134.83680725097656, 153.98580932617188, 93.34435272216797, 147.2476348876953, 89.3647232055664, 140.74720764160156, 42.561126708984375, 116.42046356201172, 92.88414001464844, 55.49193572998047, 43.27719497680664, 91.05313110351562, 34.75337600708008, 215.8608856201172, 263.023193359375, 205.83375549316406, 197.38523864746094, 203.55101013183594, 169.61834716796875, 255.87110900878906, 116.58551788330078, 96.73006439208984, 79.49942779541016, 283.06640625, 93.33574676513672, 193.91488647460938, 33.796836853027344, 262.33331298828125, 88.26288604736328, 225.51356506347656, 106.00257873535156, 183.81253051757812, 168.8426971435547, 118.38752746582031, 219.44654846191406, 253.30218505859375, 172.2328643798828, 222.41346740722656, 266.89617919921875, 297.2092590332031, 181.854736328125, 199.9833221435547, 181.0819549560547, 327.75323486328125, 233.024169921875, 247.7063446044922, 90.37926483154297, 302.8845520019531, 146.21548461914062, 217.70718383789062, 120.2440414428711, 192.33103942871094, 164.7774200439453, 181.98098754882812, 120.94720458984375, 215.16226196289062, 175.24095153808594, 235.5317840576172, 241.95535278320312, 162.02320861816406, 194.96929931640625, 225.34194946289062, 253.31613159179688, 286.50848388671875, 291.64385986328125, 284.4757385253906, 245.51397705078125, 210.78866577148438, 156.50741577148438, 133.02989196777344, 132.3738555908203, 197.6118927001953, 123.31906127929688, 143.2545623779297, 106.85369110107422, 167.89614868164062, 127.37287139892578, 153.53448486328125, 54.95347213745117, 223.73870849609375, 126.30172729492188, 223.21131896972656, 168.99034118652344, 246.63430786132812, 258.353271484375, 295.3106384277344, 284.9832458496094, 293.5976867675781, 340.20196533203125, 197.6118927001953]
a = np.array(tst, dtype=np.float32)
fast_sample(128, a)
ROWS = 20
COLS = 10
# Initialize example board and blocks
board = np.zeros((ROWS, COLS), dtype=np.int64)
blocks = np.array([0, 7, 1, 7, 2, 7, 3, 7], dtype=np.int64)  # Example blocks
N = 100000

@njit
def hard_dropp(board, blocks):
    while True:
        for i in range(4):
            if blocks[i*2]+1>=ROWS or (blocks[i*2]+1>=0 and board[blocks[i*2]+1][blocks[i*2+1]]):
                return blocks

        blocks[::2] += 1

hard_dropp(board, blocks)

@njit
def delete_lines3(board, blocks):
    for i in range(4):
        board[blocks[i*2],blocks[i*2+1]] = 1
    #board[blocks[::2], blocks[1::2]] = 1

    delete_rows = set()
    for i in range(4):
        if blocks[i*2] not in delete_rows:
            a = True
            r = blocks[i*2]
            for c in range(10):
                if board[r][c]==0:
                    a = False
                    break
            if a:
                delete_rows.add(blocks[i*2])

    if delete_rows:
        shift = len(delete_rows)
        keep_rows = np.array([r for r in range(ROWS) if r not in delete_rows])
        board[shift:] = board[keep_rows]  # Shift rows down
        board[:shift] = 0  # Zero out the top rows
    return board, len(delete_rows)

# Initialize board and blocks
ROWS, COLS = 20, 10
board = np.random.randint(0, 2, (ROWS, COLS), dtype=np.int64)  # Random board with 0s and 1s
board[5] = 1  # Fully fill row 5
board[10] = 1  # Fully fill row 10
blocks = np.array([5, 5, 5, 6, 10, 5, 10, 6], dtype=np.int64)  # Blocks in those rows
delete_lines3(board, blocks)
N = 100000

TupleType = types.UniTuple(types.int64, 6)
ValueType = types.int64
@njit
def calc_all_states3(board, rotations, r1, r2, r3, r4):
    states = Dict.empty(
        key_type=TupleType,
        value_type=ValueType
    )
    rotate_degree = 0

    for r in range(rotations):
        # rotate the piece
        if r==0:
            shape = r1
        elif r==1:
            shape = r2
        elif r==2:
            shape = r3
        elif r==3:
            shape = r4

        blocks = shape.copy()

        rotate_degree += 90

        # calculate left and right bound
        min_x = min(blocks[1::2])
        max_x = max(blocks[1::2])
        piv_x = min_x
        left_bound = piv_x - min_x
        right_bound = piv_x + 10 - max_x

        for x in range(left_bound, right_bound):
            # move each block towards x
            for i in range(4):
                blocks[i*2+1] += x - min_x

            flag = False
            for i in range(4):
                if blocks[i*2]>=0 and board[blocks[i*2], blocks[i*2+1]]>0:
                    flag = True
                    break
            if flag:
                for i in range(4):
                    blocks[i*2+1] -= x - min_x
                continue

            # copy the board and drop
            board_copy = np.copy(board)
            blocks = hard_dropp(board_copy, blocks)

            flag = True
            for i in range(4):
                if not 0<=blocks[i*2]<ROWS:
                    flag = False
                    break
            if flag:
                board_copy, lines_removed = delete_lines3(board_copy, blocks)
                x_pivot = min(blocks[1::2])
                y_pos = max(blocks[::2])
                states[get_states(board_copy, y_pos, lines_removed)] = x_pivot * 4 + r

            blocks = shape.copy()

    return states

ROWS, COLS = 20, 10
board = np.zeros((ROWS, COLS), dtype=np.int64)
blocks = np.array([0, 7, 1, 7, 2, 7, 3, 7], dtype=np.int64)
a = calc_all_states3(
    board,
    TETROMINOS['L']['rotations'],
    TETROMINOS_NUMPY['L']['shape'][0],
    TETROMINOS_NUMPY['L']['shape'][90],
    TETROMINOS_NUMPY['L']['shape'][180],
    TETROMINOS_NUMPY['L']['shape'][270]
)