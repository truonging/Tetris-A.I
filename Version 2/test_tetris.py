# import unittest
#
# from numpy.array_api import int32
#
# from tetris import Tetris
# from settings import *
# from random import randint
# from copy import deepcopy
# import numpy as np
#
# class TestTetris(unittest.TestCase):
#
#     def setUp(self):
#         pygame.init()
#         self.game = Tetris(1)
#
#     """ move x < 0 """
#     def test_move_left_out_of_bounds(self):
#         # Set piece at the left edge
#         self.game.blocks[0]['pos'] = [0, 0]  # Top-left corner
#         self.game.move_to_x(-1)
#         # Ensure it did not move out of bounds
#         self.assertEqual(self.game.blocks[0]['pos'], [0, 0], 'Failed: left oob')
#
#     """ move x > 9 """
#     def test_move_right_out_of_bounds(self):
#         # Set piece at the right edge
#         self.game.blocks[0]['pos'] = [0, COLS - 1]
#         self.game.move_to_x(10)
#         # Ensure it did not move out of bounds
#         self.assertEqual(self.game.blocks[0]['pos'], [0, COLS - 1], 'Failed: right oob')
#
#     """ bottom collision """
#     def test_bottom_collision(self):
        # Place a block at the bottom
#         self.game.blocks[0]['pos'] = [ROWS - 1, 0]
#         self.game.move_down()
#         # Ensure it did not move below the board
#         self.assertEqual(self.game.blocks[0]['pos'], [ROWS - 1, 0], 'Failed: bottom collision')
#
#     """ 1 line clear """
#     def test_clear_single_line(self):
#         self.game.board = [[0]*COLS for _ in range(ROWS)]
#         for block in self.game.blocks:
#             block['pos'] = [0,0]
#         self.game.blocks[0]['pos'][0] = ROWS-1
#         self.game.board[-1] = [1] * COLS
#         self.game.update_board(self.game.board)
#         # Check if the last line is cleared
#         self.assertEqual(self.game.board[-1], [0] * COLS, 'Failed: 1 line clear')
#
#     """ 2 line clear """
#     def test_clear_double_lines(self):
#         self.game.blocks[0]['pos'][0] = ROWS-1
#         self.game.blocks[1]['pos'][0] = ROWS-2
#         self.game.board[-1] = [1] * COLS
#         self.game.board[-2] = [1] * COLS
#         self.game.update_board(self.game.board)
#         # Check if all four lines are cleared
#         for row in range(-4, 0):
#             self.assertEqual(self.game.board[row], [0] * COLS, 'Failed: 2 line clear')
#
#     """ 4 line clear """
#     def test_clear_four_lines(self):
#         self.game.blocks[0]['pos'][0] = ROWS-1
#         self.game.blocks[1]['pos'][0] = ROWS-2
#         self.game.blocks[2]['pos'][0] = ROWS-3
#         self.game.blocks[3]['pos'][0] = ROWS-4
#         self.game.board[-1] = [1] * COLS
#         self.game.board[-2] = [1] * COLS
#         self.game.board[-3] = [1] * COLS
#         self.game.board[-4] = [1] * COLS
#         self.game.update_board(self.game.board)
#         # Check if all four lines are cleared
#         for row in range(-4, 0):
#             self.assertEqual(self.game.board[row], [0] * COLS, 'Failed: 4 line clear')
#
#     """ 2 line clear with space in between """
#     def test_clear_double_empty_lines(self):
#         self.game.blocks[0]['pos'][0] = ROWS-1
#         self.game.blocks[1]['pos'][0] = ROWS-2
#         self.game.blocks[2]['pos'][0] = ROWS-3
#         self.game.board[-1] = [1] * COLS
#         self.game.board[-2][0] = 3
#         self.game.board[-3] = [1] * COLS
#
#         self.game.update_board(self.game.board)
#
#         # Check if all 2 lines are cleared
#         self.assertEqual(self.game.board[-2], [0] * COLS, 'Failed: 2 line clear with space in between')
#         self.assertEqual(self.game.board[-3], [0] * COLS, 'Failed: 2 line clear with space in between')
#         self.assertEqual(self.game.board[-1][0], 3, 'Failed: 2 line clear with space in between')
#
#     def test_spawn_overlap(self):
#         self.game.board = [[0]*COLS for _ in range(ROWS)]
#         self.game.board[0][4] = 2
#         self.game.board[0][5] = 2
#         self.game.board[0][6] = 2
#         flag = self.game.create_tetromino()
#         self.assertEqual(flag, False, 'Failed: overlap check')
#
#     def test_spawn_oob(self):
#         self.game.board = [[0]*COLS for _ in range(ROWS)]
#         self.game.board[1][4] = 2
#         self.game.board[1][5] = 2
#         self.game.board[1][6] = 2
#         self.game.blocks[0]['pos'] = [0,4]
#         self.game.blocks[1]['pos'] = [0,5]
#         self.game.blocks[2]['pos'] = [0,6]
#         self.game.blocks[3]['pos'] = [-1,5]
#         self.game.move_down()
#
#         self.assertEqual(self.game.finished, True, 'Failed: block fall on block but pieces exist oob')
#
#     """ hard drop """
#     def test_hard_drop(self):
#         self.game.tetromino = TETROMINOS['T']
#         self.game.blocks = [{ 'color_id':self.game.tetromino['color_id'],
#                          'pos':[r,c]
#                         } for r,c in self.game.tetromino['shape'][0]
#                        ]
#         self.game.board = [[0]*COLS for _ in range(ROWS)]
#         self.game.hard_drop()
#         tst = [[19,5],[19,4],[19,6],[18,5]]
#         self.assertIn(self.game.blocks[0]['pos'], tst, 'Failed: hard drop')
#         self.assertIn(self.game.blocks[1]['pos'], tst, 'Failed: hard drop')
#         self.assertIn(self.game.blocks[2]['pos'], tst, 'Failed: hard drop')
#         self.assertIn(self.game.blocks[3]['pos'], tst, 'Failed: hard drop')
#
#     def calculate_expected(self, board, y_pos, lines_removed=0):
#
#         cols = [0] * COLS
#         holes = 0
#         bumpiness = 0
#         for c in range(len(board[0])):
#             block = False
#             for r in range(len(board)):
#                 if board[r][c] > 0 and not block:
#                     block = True
#                     cols[c] = 20 - r
#                 if not board[r][c] and block:
#                     holes += 1
#             if c > 0:
#                 bumpiness += abs(cols[c] - cols[c - 1])
#         total_heights = sum(cols)
#         # bumpiness = sum(abs(cols[i ] -cols[ i -1]) for i in range(1 ,len(cols)))
#         # pillar = any(cols[i-1]-cols[i]>=3 and cols[i+1]-cols[i]>=3 for i in range(1,len(cols)-1)) or cols[1]-cols[0]>=3 or cols[-1]-cols[-2]>=3
#         pillar = 0
#
#         # Check pillars in the middle columns
#         for i in range(1, len(cols) - 1):
#             if (cols[i - 1] - cols[i] >= 3) and (cols[i + 1] - cols[i] >= 3):
#                 pillar = 1
#                 break
#
#         # Check edge cases for the first and last columns
#         if pillar == 0:
#             if cols[1] - cols[0] >= 3 or cols[-2] - cols[-1] >= 3:
#                 pillar = 1
#
#         y_pos = y_pos
#
#         state = [total_heights, bumpiness, lines_removed, holes, [], pillar]
#
#         return state
#
#     def generate_random_board(self):
#         return [[randint(0, 1) for _ in range(COLS)] for _ in range(ROWS)]
#
#     def test_random_boards(self):
#         for _ in range(10):  # Run 10 random tests
#             board = self.generate_random_board()
#             self.game.board = deepcopy(board)
#             self.game.blocks = []  # Assuming no falling blocks
#
#             expected = self.calculate_expected(board, self.game.blocks)
#             y_pos = 0
#             board = np.array(board, dtype=np.int32)
#             actual = get_states(board, y_pos)
#             actual[4] = expected[4]
#             self.assertEqual(actual, expected, f"Failed on board: {board}")
#
#     def test_edge_cases(self):
#         # Empty board
#         empty_board = [[0] * COLS for _ in range(ROWS)]
#         self.game.board = deepcopy(empty_board)
#         self.game.blocks = []
#         expected = self.calculate_expected(empty_board, self.game.blocks)
#         y_pos = 0
#         empty_board = np.array(empty_board, dtype=np.int32)
#         actual = get_states(empty_board, y_pos)
#         actual[4] = expected[4]
#         self.assertEqual(actual, expected, "Failed on empty board")
#
#         # Full board
#         full_board = [[1] * COLS for _ in range(ROWS)]
#         self.game.board = deepcopy(full_board)
#         self.game.blocks = []
#         expected = self.calculate_expected(full_board, self.game.blocks)
#         y_pos = 0
#         full_board = np.array(full_board, dtype=int32)
#         actual = get_states(full_board, y_pos)
#         actual[4] = expected[4]
#         self.assertEqual(actual, expected, "Failed on full board")
#
#         # Staggered columns
#         staggered_board = [
#             [0, 0, 0, 0],
#             [1, 0, 0, 0],
#             [1, 1, 0, 0],
#             [1, 1, 1, 0],
#             [1, 1, 1, 1]
#         ]
#         self.game.board = deepcopy(staggered_board)
#         self.game.blocks = []
#         expected = self.calculate_expected(staggered_board, self.game.blocks)
#         y_pos = 0
#         staggered_board = np.array(staggered_board, dtype=np.int32)
#         actual = get_states(staggered_board, y_pos)
#         actual[4] = expected[4]
#         self.assertEqual(actual, expected, "Failed on staggered board")
#
#     def test_falling_blocks(self):
#         # Board with falling blocks
#         board = [
#             [0, 0, 0, 0],
#             [1, 0, 0, 0],
#             [1, 1, 0, 0],
#             [1, 1, 1, 0],
#             [1, 1, 1, 1]
#         ]
#         self.game.board = deepcopy(board)
#         a = pygame.sprite.LayeredDirty()
#         self.game.blocks = [
#             {'color_id': 1, 'pos': [1,2]},
#             {'color_id': 1, 'pos': [2, 3]}
#         ]
#         expected = self.calculate_expected(board, self.game.blocks)
#         y_pos = 0
#         board = np.array(board, dtype=np.int32)
#         actual = get_states(board, 0)
#         actual[4] = expected[4]
#         self.assertEqual(actual, expected, "Failed with falling blocks")
#
#
# class TestCalcAllStates(unittest.TestCase):
#
#     def setUp(self):
#         pygame.init()
#         self.game = Tetris(1)
#         self.game.board = [[0] * COLS for _ in range(ROWS)]
#         self.game.tetromino = TETROMINOS['I']  # 'I' piece has 2 unique rotations
#         a = pygame.sprite.LayeredDirty()
#
#         self.game.blocks = [ # vertical I
#             {'color_id': 2, 'pos': [0, 4]},
#             {'color_id': 2, 'pos': [1, 4]},
#             {'color_id': 2, 'pos': [2, 4]},
#             {'color_id': 2, 'pos': [3, 4]}
#         ]
#
#     def test_calc_all_states_I_piece(self):
#         states = self.game.calc_all_states()
#
#         # Expected number of states:
#         # - 'I' piece has 2 rotations (vertical and horizontal)
#         # - Vertical: Can drop in columns 0 to 9
#         # - Horizontal: Can drop in columns 0 to 6 (since it occupies 4 columns)
#         expected_num_states = (COLS) + (COLS - 3)
#
#         self.assertEqual(len(states), expected_num_states,
#                          f"Expected {expected_num_states} states, but got {len(states)}.")
#
#         # Check that all states are valid
#         for (x_pivot, rotation), state in states.items():
#             self.assertTrue(0 <= x_pivot < COLS, f"Invalid x_pivot: {x_pivot}")
#             self.assertIn(rotation, [0, 1], f"Invalid rotation: {rotation}")
#             self.assertEqual(len(state), 6, f"State should have 6 features, but got {len(state)}")

        # Optional: Print states for visual inspection
        # for pos, state in states.items():
        #     print(f"Position: {pos}, State: {state}")



# if __name__ == "__main__":
#     unittest.main()


"""
         201445791 function calls (189861647 primitive calls) in 314.535 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    1.593    1.593  312.872  312.872 train.py:81(run_simulation)
    36316    9.686    0.000  169.194    0.005 game.py:213(calc_all_states)
    36096    0.148    0.000   92.559    0.003 tetris.py:68(play_full)
    70946    0.316    0.000   91.785    0.001 tetris.py:125(play_step)
   855277    3.647    0.000   76.758    0.000 game.py:322(hard_drop)
  8832446   12.835    0.000   73.112    0.000 game.py:327(move_down)
    70946    0.161    0.000   65.083    0.001 game.py:260(run)
   797388    9.561    0.000   58.899    0.000 game.py:247(get_state)
 42803289   27.949    0.000   47.628    0.000 game.py:332(<genexpr>)
    70946    1.180    0.000   47.328    0.001 game.py:76(_display_game)
   797388   11.212    0.000   30.792    0.000 njit_startup.py:17(get_states_fast)
      503    0.003    0.000   25.317    0.050 agent.py:79(train_long_memory)
      503    0.050    0.000   25.312    0.050 model.py:99(train)
     1006    0.213    0.000   25.262    0.025 model.py:65(train_step)
   496622   23.170    0.000   23.170    0.000 {method 'blit' of 'pygame.surface.Surface' objects}
   359511   20.816    0.000   20.816    0.000 {method 'fill' of 'pygame.surface.Surface' objects}
      988    0.757    0.001   19.902    0.020 prioritized_memory.py:45(update_priority)
 33970843   19.679    0.000   19.679    0.000 game.py:330(<lambda>)
      988   19.032    0.019   19.032    0.019 {built-in method _heapq.heapify}
538648/108520    0.679    0.000   16.409    0.000 module.py:1735(_wrapped_call_impl)
538648/108520    0.952    0.000   16.240    0.000 module.py:1743(_call_impl)
   107532    1.113    0.000   15.756    0.000 model.py:16(forward)
    36096    1.018    0.000   15.416    0.000 agent.py:57(remember)
  3241199    5.151    0.000   15.170    0.000 fromnumeric.py:71(_wrapreduction)
  2392164    2.785    0.000   13.750    0.000 fromnumeric.py:2177(sum)
    70946    0.111    0.000   13.059    0.000 game.py:97(_input)
    30/20    0.000    0.000   12.001    0.600 _ops.py:291(fallthrough)
   975814    2.064    0.000   11.036    0.000 game.py:293(move_to_x)
   430128    0.556    0.000   10.395    0.000 linear.py:124(forward)
   430128    9.539    0.000    9.539    0.000 {built-in method torch._C._nn.linear}
  3243175    9.227    0.000    9.227    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    71000    9.176    0.000    9.176    0.000 {built-in method pygame.display.update}
    36042    0.556    0.000    8.828    0.000 game.py:144(_update_board)
  1597740    1.317    0.000    8.036    0.000 fromnumeric.py:53(_wrapfunc)
    36096    0.669    0.000    7.976    0.000 agent.py:89(get_action)
    70946    0.252    0.000    7.763    0.000 sprite.py:558(draw)
  1986488    7.118    0.000    7.118    0.000 {built-in method pygame.draw.line}
    70946    5.836    0.000    7.014    0.000 {method 'blits' of 'pygame.surface.Surface' objects}
   797388    5.365    0.000    6.349    0.000 function_base.py:1324(diff)
    70946    0.862    0.000    6.032    0.000 scoreboard.py:37(run)
"""

"""
         118771244 function calls (109559612 primitive calls) in 168.302 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.755    0.755  166.649  166.649 main_screen.py:35(run)
    27585    0.381    0.000   77.297    0.003 main_screen.py:23(calculate_action)
    27585    4.236    0.000   65.221    0.002 tetris.py:80(calc_all_states)
    27545    0.253    0.000   51.206    0.002 tetris.py:193(play)
    27545    0.123    0.000   45.600    0.002 game_screen.py:20(draw)
    27545    0.117    0.000   37.086    0.001 game_screen.py:76(draw)
      637    0.002    0.000   28.467    0.045 Agent.py:86(train_long_memory)
      637    0.063    0.000   28.463    0.045 model.py:95(train)
     1274    0.201    0.000   28.400    0.022 model.py:61(train_step)
   629591    2.214    0.000   25.311    0.000 tetris.py:155(hard_drop)
   617323   15.360    0.000   25.071    0.000 tetris.py:115(get_states)
  7153777    6.688    0.000   24.351    0.000 tetris.py:160(move_down)
     1266    0.971    0.001   22.113    0.017 Agent.py:247(update_priority)
   330540   21.220    0.000   21.220    0.000 {method 'blit' of 'pygame.surface.Surface' objects}
     1266   20.991    0.017   20.991    0.017 {built-in method _heapq.heapify}
    27545    1.868    0.000   15.662    0.001 game_screen.py:68(draw_board)
  2099803   15.643    0.000   15.643    0.000 {built-in method pygame.draw.rect}
417776/84568    0.522    0.000   12.180    0.000 module.py:1735(_wrapped_call_impl)
417776/84568    0.696    0.000   12.049    0.000 module.py:1743(_call_impl)
    30/20    0.000    0.000   11.961    0.598 _ops.py:291(fallthrough)
    27545    0.485    0.000   11.674    0.000 Agent.py:97(get_action)
    83302    0.861    0.000   11.665    0.000 model.py:12(forward)
    27545    0.649    0.000   11.126    0.000 Agent.py:64(remember)
 34735476    9.996    0.000    9.996    0.000 tetris.py:164(<genexpr>)
   333208    0.385    0.000    7.517    0.000 linear.py:124(forward)
   333208    6.892    0.000    6.892    0.000 {built-in method torch._C._nn.linear}
   644868    0.808    0.000    6.351    0.000 tetris.py:56(update_board)
"""

"""
         131845361 function calls (121434184 primitive calls) in 194.850 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.834    0.834  193.254  193.254 main_screen.py:35(run)
    30026    0.427    0.000   83.845    0.003 main_screen.py:23(calculate_action)
    30026    4.552    0.000   70.052    0.002 tetris.py:80(calc_all_states)
    30001    0.280    0.000   69.166    0.002 tetris.py:193(play)
    30001    0.072    0.000   63.049    0.002 game_screen.py:23(draw)
   330011   40.210    0.000   40.210    0.000 {method 'blit' of 'pygame.surface.Surface' objects}
    30001    0.151    0.000   38.187    0.001 game_screen.py:72(draw)
      648    0.003    0.000   29.427    0.045 Agent.py:86(train_long_memory)
      648    0.065    0.000   29.422    0.045 model.py:95(train)
     1296    0.203    0.000   29.356    0.023 model.py:61(train_step)
   684245    2.419    0.000   27.734    0.000 tetris.py:155(hard_drop)
  8172765    7.336    0.000   26.683    0.000 tetris.py:160(move_down)
   676720   16.293    0.000   26.580    0.000 tetris.py:115(get_states)
     1288    1.004    0.001   22.753    0.018 Agent.py:247(update_priority)
     1288   21.594    0.017   21.594    0.017 {built-in method _heapq.heapify}
    30001    1.039    0.000   14.884    0.000 game_screen.py:66(draw_board)
  2088770   14.782    0.000   14.782    0.000 {built-in method pygame.draw.rect}
453783/91787    0.566    0.000   13.639    0.000 module.py:1735(_wrapped_call_impl)
453783/91787    0.768    0.000   13.501    0.000 module.py:1743(_call_impl)
    30001    0.556    0.000   13.342    0.000 Agent.py:97(get_action)
    90499    0.934    0.000   13.063    0.000 model.py:12(forward)
    30001    0.722    0.000   12.368    0.000 Agent.py:64(remember)
    30/20    0.000    0.000   12.034    0.602 _ops.py:291(fallthrough)
 39742911   11.038    0.000   11.038    0.000 tetris.py:164(<genexpr>)
   361996    0.405    0.000    8.541    0.000 linear.py:124(forward)
   361996    7.882    0.000    7.882    0.000 {built-in method torch._C._nn.linear}
   706721    0.875    0.000    6.859    0.000 tetris.py:56(update_board)
   182011    6.063    0.000    6.063    0.000 {method 'fill' of 'pygame.surface.Surface' objects}
"""

"""
         133317803 function calls (122591090 primitive calls) in 199.383 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.845    0.845  197.780  197.780 main_screen.py:35(run)
    30625    0.418    0.000   86.938    0.003 main_screen.py:23(calculate_action)
    30625    4.646    0.000   73.216    0.002 tetris.py:80(calc_all_states)
    30595    0.287    0.000   70.045    0.002 tetris.py:193(play)
    30595    0.078    0.000   63.845    0.002 game_screen.py:23(draw)
   214165   42.173    0.000   42.173    0.000 {method 'blit' of 'pygame.surface.Surface' objects}
    30595    0.099    0.000   37.645    0.001 game_screen.py:83(draw)
      651    0.002    0.000   30.042    0.046 Agent.py:86(train_long_memory)
      651    0.066    0.000   30.037    0.046 model.py:95(train)
     1302    0.199    0.000   29.972    0.023 model.py:61(train_step)
   696376    2.559    0.000   29.459    0.000 tetris.py:155(hard_drop)
  8451971    7.719    0.000   28.352    0.000 tetris.py:160(move_down)
   686205   16.858    0.000   27.529    0.000 tetris.py:115(get_states)
     1290    1.009    0.001   23.608    0.018 Agent.py:247(update_priority)
     1290   22.444    0.017   22.444    0.017 {built-in method _heapq.heapify}
    30595    1.064    0.000   14.855    0.000 game_screen.py:70(draw_board)
  2085406   14.781    0.000   14.781    0.000 {built-in method pygame.draw.rect}
462950/93622    0.574    0.000   13.621    0.000 module.py:1735(_wrapped_call_impl)
462950/93622    0.794    0.000   13.481    0.000 module.py:1743(_call_impl)
    30595    0.545    0.000   13.281    0.000 Agent.py:97(get_action)
    92332    0.973    0.000   13.055    0.000 model.py:12(forward)
    30595    0.755    0.000   12.608    0.000 Agent.py:64(remember)
    30/20    0.000    0.000   12.027    0.601 _ops.py:291(fallthrough)
 41113332   11.703    0.000   11.703    0.000 tetris.py:164(<genexpr>)
   369328    0.426    0.000    8.400    0.000 linear.py:124(forward)
   369328    7.709    0.000    7.709    0.000 {built-in method torch._C._nn.linear}
   716800    0.894    0.000    6.924    0.000 tetris.py:56(update_board)
"""
"""
         92377296 function calls (83538119 primitive calls) in 159.336 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.735    0.735  157.687  157.687 main_screen.py:35(run)
    28767    0.248    0.000   63.309    0.002 tetris.py:196(play)
    28810    0.354    0.000   57.996    0.002 main_screen.py:23(calculate_action)
    28767    0.062    0.000   57.960    0.002 game_screen.py:23(draw)
    28810    5.473    0.000   46.772    0.002 tetris.py:80(calc_all_states)
   201369   37.482    0.000   37.482    0.000 {method 'blit' of 'pygame.surface.Surface' objects}
    28767    0.085    0.000   34.776    0.001 game_screen.py:70(draw)
      643    0.002    0.000   26.768    0.042 Agent.py:86(train_long_memory)
      643    0.055    0.000   26.764    0.042 model.py:95(train)
     1286    0.114    0.000   26.709    0.021 model.py:61(train_step)
   653538    2.118    0.000   22.332    0.000 tetris.py:157(hard_drop)
  7340812    6.327    0.000   21.375    0.000 tetris.py:162(move_down)
     1272    0.934    0.001   21.218    0.017 Agent.py:247(update_priority)
     1272   20.141    0.016   20.141    0.016 {built-in method _heapq.heapify}
    28767    0.996    0.000   14.416    0.001 game_screen.py:64(draw_board)
  2054753   14.275    0.000   14.275    0.000 {built-in method pygame.draw.rect}
    30/20    0.000    0.000   12.662    0.633 _ops.py:291(fallthrough)
436647/88347    0.515    0.000   11.427    0.000 module.py:1735(_wrapped_call_impl)
436647/88347    0.703    0.000   11.304    0.000 module.py:1743(_call_impl)
    87075    0.851    0.000   10.945    0.000 model.py:12(forward)
    28767    0.668    0.000   10.908    0.000 Agent.py:64(remember)
    28767    0.463    0.000   10.850    0.000 Agent.py:97(get_action)
   637245    8.373    0.000    8.373    0.000 {built-in method numpy.array}
 35584046    7.936    0.000    7.936    0.000 tetris.py:167(<genexpr>)
   348300    0.371    0.000    6.874    0.000 linear.py:124(forward)
   348300    6.288    0.000    6.288    0.000 {built-in method torch._C._nn.linear}
   639830    0.715    0.000    5.917    0.000 tetris.py:56(update_board)
   174607    5.405    0.000    5.405    0.000 {method 'fill' of 'pygame.surface.Surface' objects}
"""

"""
         80645046 function calls (72729607 primitive calls) in 136.615 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.683    0.683  134.927  134.927 main_screen.py:35(run)
    24571    0.336    0.000   55.012    0.002 main_screen.py:23(calculate_action)
    24571    5.164    0.000   44.533    0.002 tetris.py:81(calc_all_states)
    24545    0.239    0.000   43.933    0.002  .py:199(play)
    24545    0.093    0.000   38.933    0.002 game_screen.py:23(draw)
   448799   34.718    0.000   34.718    0.000 {method 'blit' of 'pygame.surface.Surface' objects}
      622    0.003    0.000   26.697    0.043 Agent.py:86(train_long_memory)
      622    0.064    0.000   26.692    0.043 model.py:96(train)
     1244    0.129    0.000   26.628    0.021 model.py:61(train_step)
   560464    2.037    0.000   21.892    0.000 tetris.py:158(hard_drop)
  6625134    6.181    0.000   20.975    0.000 tetris.py:165(move_down)
     1230    0.928    0.001   20.402    0.017 Agent.py:247(update_priority)
     1230   19.329    0.016   19.329    0.016 {built-in method _heapq.heapify}
    24545    0.863    0.000   17.154    0.001 game_screen.py:92(draw)
    30/20    0.000    0.000   13.029    0.651 _ops.py:291(fallthrough)
373695/75723    0.474    0.000   11.205    0.000 module.py:1735(_wrapped_call_impl)
373695/75723    0.642    0.000   11.090    0.000 module.py:1743(_call_impl)
    74493    0.783    0.000   10.732    0.000 model.py:12(forward)
    24545    0.433    0.000   10.124    0.000 Agent.py:97(get_action)
    24545    0.618    0.000   10.079    0.000 Agent.py:64(remember)
 32172182    7.714    0.000    7.714    0.000 tetris.py:170(<genexpr>)
   549710    7.692    0.000    7.692    0.000 {built-in method numpy.array}
   297972    0.338    0.000    6.971    0.000 linear.py:124(forward)
   297972    6.422    0.000    6.422    0.000 {built-in method torch._C._nn.linear}
"""

"""
         218861151 function calls (195930764 primitive calls) in 194.867 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        2    2.062    1.031  192.822   96.411 main_screen.py:89(run)
    72354    0.974    0.000  138.348    0.002 main_screen.py:78(play_action)
    72354   14.143    0.000  118.482    0.002 tetris.py:85(calc_all_states)
  1573892    5.396    0.000   58.053    0.000 tetris.py:129(hard_drop)
 19202740   16.297    0.000   55.475    0.000 tetris.py:139(move_down)
1085369/219421    1.371    0.000   30.765    0.000 module.py:1735(_wrapped_call_impl)
1085369/219421    1.959    0.000   30.438    0.000 module.py:1743(_call_impl)
   216487    2.047    0.000   29.518    0.000 model.py:12(forward)
    72285    1.536    0.000   25.882    0.000 Agent.py:66(remember)
 93408339   20.073    0.000   20.073    0.000 tetris.py:143(<genexpr>)
  1530859   19.502    0.000   19.502    0.000 {built-in method numpy.array}
   865948    0.942    0.000   19.190    0.000 linear.py:124(forward)
   865948   17.581    0.000   17.581    0.000 {built-in method torch._C._nn.linear}
     2000    0.005    0.000   14.597    0.007 Agent.py:88(train_long_memory)
     2000    0.138    0.000   14.586    0.007 model.py:95(train)
     4000    0.363    0.000   14.447    0.004 model.py:61(train_step)
    31/21    0.000    0.000   14.002    0.667 _ops.py:291(fallthrough)
    72285    1.358    0.000   13.443    0.000 Agent.py:98(get_action)
  1742739    2.624    0.000   11.064    0.000 tetris.py:162(move_to_x)
  3405288    4.464    0.000    7.158    0.000 {built-in method builtins.min}
   576659    7.075    0.000    7.075    0.000 {method 'blit' of 'pygame.surface.Surface' objects}
     2934    0.036    0.000    6.201    0.002 model.py:48(fit)
  1538098    1.501    0.000    5.682    0.000 tetris.py:63(update_board)
    72285    0.628    0.000    5.398    0.000 tetris.py:182(play)
     2934    2.117    0.001    4.779    0.002 Agent.py:198(update_priority)
  7742570    2.984    0.000    4.559    0.000 {built-in method builtins.all}
       89    0.002    0.000    4.216    0.047 __init__.py:1(<module>)
  1538098    2.663    0.000    4.181    0.000 tetris.py:53(delete_lines)
   649461    0.534    0.000    4.083    0.000 functional.py:1693(relu)
  1665721    2.615    0.000    3.864    0.000 {built-in method builtins.max}
   649461    3.404    0.000    3.404    0.000 {built-in method torch.relu}
    25/20    0.000    0.000    3.401    0.170 _ops.py:279(py_impl)
  8526497    2.731    0.000    2.731    0.000 tetris.py:165(<genexpr>)
     2934    2.333    0.001    2.333    0.001 {built-in method _heapq.heapify}
        3    0.000    0.000    2.261    0.754 _ops.py:147(py_functionalize_impl)
     2934    0.040    0.000    2.104    0.001 optimizer.py:473(wrapper)
     2934    0.009    0.000    1.948    0.001 _tensor.py:570(backward)
     2934    0.017    0.000    1.938    0.001 __init__.py:242(backward)
     2265    0.151    0.000    1.931    0.001 game_screen.py:277(draw)
     2934    0.024    0.000    1.861    0.001 optimizer.py:72(_use_grad)
     2934    0.012    0.000    1.845    0.001 graph.py:814(_engine_run_backward)
     2934    1.830    0.001    1.830    0.001 {method 'run_backward' of 'torch._C._EngineBase' objects}
     2934    0.022    0.000    1.826    0.001 adam.py:210(step)
     2265    1.773    0.001    1.773    0.001 {built-in method pygame.display.update}
    72285    0.341    0.000    1.717    0.000 Agent.py:165(add)
   157106    1.617    0.000    1.617    0.000 {built-in method torch.tensor}
     5354    1.597    0.000    1.597    0.000 {method 'fill' of 'pygame.surface.Surface' objects}
"""
"""
         66325407 function calls (61749902 primitive calls) in 127.280 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        2    1.965    0.982  125.247   62.624 main_screen.py:114(run)
    71567    0.939    0.000   73.671    0.001 main_screen.py:78(play_action)
    71567   19.421    0.000   54.284    0.001 tetris.py:199(calc_all_states3)
1070698/216442    1.329    0.000   29.005    0.000 module.py:1735(_wrapped_call_impl)
1070698/216442    1.935    0.000   28.683    0.000 module.py:1743(_call_impl)
   213564    2.053    0.000   27.807    0.000 model.py:12(forward)
    71500    1.523    0.000   25.565    0.000 Agent.py:66(remember)
   854256    0.901    0.000   17.464    0.000 linear.py:124(forward)
   854256   15.881    0.000   15.881    0.000 {built-in method torch._C._nn.linear}
    31/21    0.000    0.000   14.050    0.669 _ops.py:291(fallthrough)
     2100    0.005    0.000   13.199    0.006 Agent.py:88(train_long_memory)
     2100    0.121    0.000   13.188    0.006 model.py:95(train)
     4200    0.246    0.000   13.066    0.003 model.py:61(train_step)
    71500    1.299    0.000   12.072    0.000 Agent.py:98(get_action)
  1452050    9.255    0.000   11.413    0.000 tetris.py:184(delete_lines3) 
   559074    7.165    0.000    7.165    0.000 {method 'blit' of 'pygame.surface.Surface' objects}
  7671243    3.736    0.000    6.576    0.000 {built-in method builtins.all}
    71500    0.584    0.000    6.326    0.000 tetris.py:139(play)
     2878    0.031    0.000    5.576    0.002 model.py:48(fit)
  8062853    4.796    0.000    4.796    0.000 tetris.py:223(<genexpr>) 
     2878    2.094    0.001    4.541    0.002 Agent.py:198(update_priority)
       89    0.002    0.000    4.209    0.047 __init__.py:1(<module>)
   640692    0.504    0.000    4.111    0.000 functional.py:1693(relu)
  1703752    3.875    0.000    3.875    0.000 {built-in method numpy.array}
   848148    1.244    0.000    3.532    0.000 tetris.py:108(move_down)
   640692    3.454    0.000    3.454    0.000 {built-in method torch.relu}
    25/20    0.000    0.000    3.413    0.171 _ops.py:279(py_impl)
  1717541    3.081    0.000    3.081    0.000 {built-in method builtins.min}
  7619591    2.839    0.000    2.839    0.000 tetris.py:231(<genexpr>)
  1649492    2.529    0.000    2.532    0.000 {built-in method builtins.max}
  1557879    0.574    0.000    2.315    0.000 function_base.py:873(copy)
        3    0.000    0.000    2.269    0.756 _ops.py:147(py_functionalize_impl)
"""

"""
         33466648 function calls (30410748 primitive calls) in 82.904 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        2    2.023    1.012   80.873   40.436 main_screen.py:136(run)
1123845/227049    1.323    0.000   29.144    0.000 module.py:1735(_wrapped_call_impl)
    75023    5.193    0.000   29.092    0.000 main_screen.py:78(play_action)
1123845/227049    1.938    0.000   28.834    0.000 module.py:1743(_call_impl)
   224199    2.120    0.000   27.959    0.000 model.py:12(forward)
    74953    1.555    0.000   25.814    0.000 Agent.py:66(remember)
   896796    1.064    0.000   17.611    0.000 linear.py:124(forward)
   896796   15.844    0.000   15.844    0.000 {built-in method torch._C._nn.linear}
    31/21    0.000    0.000   14.951    0.712 _ops.py:291(fallthrough)
    74953    1.810    0.000   14.694    0.000 Agent.py:98(get_action)
     2000    0.005    0.000   13.452    0.007 Agent.py:88(train_long_memory)
     2000    0.119    0.000   13.441    0.007 model.py:95(train)
     4000    0.226    0.000   13.322    0.003 model.py:61(train_step)
   596136    6.902    0.000    6.902    0.000 {method 'blit' of 'pygame.surface.Surface' objects}
    74953    0.590    0.000    6.104    0.000 tetris.py:139(play)
     2850    0.032    0.000    5.637    0.002 model.py:48(fit)
       89    0.002    0.000    4.884    0.055 __init__.py:1(<module>)
     2850    2.101    0.001    4.776    0.002 Agent.py:198(update_priority)
   672597    0.524    0.000    4.061    0.000 functional.py:1693(relu)
    25/20    0.000    0.000    3.635    0.182 _ops.py:279(py_impl)
   887257    1.253    0.000    3.507    0.000 tetris.py:108(move_down)
   672597    3.393    0.000    3.393    0.000 {built-in method torch.relu}
        3    0.000    0.000    2.417    0.806 _ops.py:147(py_functionalize_impl)
     2850    2.345    0.001    2.345    0.001 {built-in method _heapq.heapify}
   142696    2.215    0.000    2.215    0.000 {built-in method numpy.array}
     2850    0.036    0.000    1.981    0.001 optimizer.py:473(wrapper)
     2245    0.144    0.000    1.832    0.001 game_screen.py:277(draw)
     2850    0.022    0.000    1.766    0.001 optimizer.py:72(_use_grad)
     2850    0.019    0.000    1.732    0.001 adam.py:210(step)
     2850    0.009    0.000    1.712    0.001 _tensor.py:570(backward)
    74953    0.339    0.000    1.708    0.000 Agent.py:165(add)
     2850    0.016    0.000    1.703    0.001 __init__.py:242(backward)
     2245    1.674    0.001    1.674    0.001 {built-in method pygame.display.update}
     2850    0.010    0.000    1.615    0.001 graph.py:814(_engine_run_backward)
   162106    1.611    0.000    1.611    0.000 {built-in method torch.tensor}
     2850    1.603    0.001    1.603    0.001 {method 'run_backward' of 'torch._C._EngineBase' objects}
2485271/2485260    0.435    0.000    1.600    0.000 <frozen _collections_abc>:868(__iter__)
     2850    0.008    0.000    1.499    0.001 optimizer.py:140(maybe_fallback)
     2850    0.014    0.000    1.489    0.001 adam.py:809(adam)
  2699988    1.463    0.000    1.463    0.000 module.py:1915(__getattr__)
     2850    0.556    0.000    1.418    0.000 adam.py:340(_single_tensor_adam)
80687/80670    0.088    0.000    1.331    0.000 _tensor.py:33(wrapped)
        3    0.000    0.000    1.221    0.407 _ops.py:251(__init__)
        1    0.000    0.000    1.196    1.196 triton_kernel_wrap.py:650(__init__)
  1123846    1.195    0.000    1.195    0.000 {built-in method torch._C._get_tracing_state}
  4287090    1.179    0.000    1.179    0.000 tetris.py:112(<genexpr>)
    77805    0.052    0.000    1.174    0.000 _tensor.py:1071(__rsub__)
   139776    0.731    0.000    1.165    0.000 typeddict.py:192(__iter__)
    77805    1.122    0.000    1.122    0.000 {built-in method torch.rsub}
"""

"""
         32334852 function calls (29439765 primitive calls) in 72.244 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        2    0.830    0.415   70.250   35.125 main_screen.py:136(run)
    70666    5.051    0.000   27.929    0.000 main_screen.py:78(play_action)
1061103/214491    1.278    0.000   25.396    0.000 module.py:1735(_wrapped_call_impl)
1061103/214491    1.873    0.000   25.094    0.000 module.py:1743(_call_impl)
   211653    2.072    0.000   24.257    0.000 model.py:12(forward)
    70610    1.730    0.000   20.537    0.000 Agent.py:66(remember)
   846612    0.893    0.000   14.673    0.000 linear.py:124(forward)
    31/21    0.000    0.000   14.524    0.692 _ops.py:291(fallthrough)
    70610    1.718    0.000   13.995    0.000 Agent.py:103(get_action)
   846612   13.124    0.000   13.124    0.000 {built-in method torch._C._nn.linear}
     2000    0.005    0.000   10.894    0.005 Agent.py:93(train_long_memory)
     2000    0.117    0.000   10.883    0.005 model.py:95(train)
     4000    0.229    0.000   10.766    0.003 model.py:61(train_step)
   561649    6.582    0.000    6.582    0.000 {method 'blit' of 'pygame.surface.Surface' objects}
    70610    0.559    0.000    5.929    0.000 tetris.py:139(play)
     2838    0.030    0.000    5.442    0.002 model.py:48(fit)
       89    0.002    0.000    4.732    0.053 __init__.py:1(<module>)
    25/20    0.000    0.000    3.529    0.176 _ops.py:279(py_impl)
   634959    0.493    0.000    3.516    0.000 functional.py:1693(relu)
   840097    1.196    0.000    3.352    0.000 tetris.py:108(move_down)
   634959    2.890    0.000    2.890    0.000 {built-in method torch.relu}
     2838    2.024    0.001    2.472    0.001 Agent.py:203(update_priority)
        3    0.000    0.000    2.346    0.782 _ops.py:147(py_functionalize_impl)
   134529    2.094    0.000    2.094    0.000 {built-in method numpy.array}
     2838    0.035    0.000    1.938    0.001 optimizer.py:473(wrapper)
     2138    0.136    0.000    1.752    0.001 game_screen.py:277(draw)
     2838    0.021    0.000    1.727    0.001 optimizer.py:72(_use_grad)
     2838    0.019    0.000    1.695    0.001 adam.py:210(step)
     2838    0.008    0.000    1.673    0.001 _tensor.py:570(backward)
     2838    0.015    0.000    1.663    0.001 __init__.py:242(backward)
     2138    1.654    0.001    1.654    0.001 {built-in method pygame.display.update}
2336288/2336277    0.415    0.000    1.603    0.000 <frozen _collections_abc>:868(__iter__)
     2838    0.009    0.000    1.579    0.001 graph.py:814(_engine_run_backward)
     2838    1.567    0.001    1.567    0.001 {method 'run_backward' of 'torch._C._EngineBase' objects}
     2838    0.008    0.000    1.470    0.001 optimizer.py:140(maybe_fallback)
     2838    0.014    0.000    1.460    0.001 adam.py:809(adam)
  2549436    1.415    0.000    1.415    0.000 module.py:1915(__getattr__)
     2838    0.547    0.000    1.391    0.000 adam.py:340(_single_tensor_adam)
"""

"""
         223807016 function calls (210035110 primitive calls) in 483.520 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    2.155    2.155  481.741  481.741 train.py:85(run_simulation)
    42254    0.305    0.000  206.318    0.005 tetris.py:87(play_full)
    82161    0.575    0.000  205.107    0.002 tetris.py:139(play_step)
    42383   12.129    0.000  180.555    0.004 game.py:203(calc_all_states)
    82160    0.253    0.000  114.913    0.001 game.py:250(run)
  1005317    4.698    0.000   96.764    0.000 game.py:311(hard_drop)
 10562398   16.251    0.000   92.066    0.000 game.py:316(move_down)
    82160    1.534    0.000   91.179    0.001 game.py:66(_display_game)
  2464800   63.392    0.000   63.392    0.000 {method 'blit' of 'pygame.surface.Surface' objects}
 51224712   35.103    0.000   59.718    0.000 game.py:321(<genexpr>)
      501    0.002    0.000   57.006    0.114 agent.py:127(train_long_memory)
      501    0.061    0.000   57.001    0.114 model.py:88(train)
     1002    0.264    0.000   56.940    0.057 model.py:54(train_step)
      980    0.761    0.001   48.374    0.049 prioritized_memory.py:43(update_priority)
    82160    4.287    0.000   48.362    0.001 scoreboard.py:52(run)
      980   47.498    0.048   47.498    0.048 {built-in method _heapq.heapify}
   416948   47.345    0.000   47.345    0.000 {method 'fill' of 'pygame.surface.Surface' objects}
   940168   18.161    0.000   42.586    0.000 game.py:237(get_state)
 40662314   24.615    0.000   24.615    0.000 game.py:319(<lambda>)
627665/126317    0.944    0.000   23.248    0.000 module.py:1735(_wrapped_call_impl)
627665/126317    1.365    0.000   22.975    0.000 module.py:1743(_call_impl)
    42253    1.508    0.000   22.695    0.001 agent.py:105(remember)
   125337    1.520    0.000   22.196    0.000 model.py:16(forward)
    82411   21.174    0.000   21.174    0.000 {built-in method pygame.display.update}
  2218320    2.105    0.000   20.814    0.000 scoreboard.py:47(display_text)
    82160    0.168    0.000   17.861    0.000 game.py:87(_input)
   501348    0.710    0.000   14.935    0.000 linear.py:124(forward)
    82160    0.365    0.000   14.013    0.000 sprite.py:558(draw)
    30/20    0.000    0.000   13.904    0.695 _ops.py:291(fallthrough)
   501348   13.810    0.000   13.810    0.000 {built-in method torch._C._nn.linear}
  1136723    2.520    0.000   13.328    0.000 game.py:283(move_to_x)
    82160   11.216    0.000   12.910    0.000 {method 'blits' of 'pygame.surface.Surface' objects}
"""

"""
         22190082 function calls (20214157 primitive calls) in 52.530 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.423    0.423   50.491   50.491 main_screen.py:155(run2)
    46619    3.519    0.000   20.576    0.000 main_screen.py:113(play_action)
    31/21    0.000    0.000   17.968    0.856 _ops.py:291(fallthrough)
698733/140673    0.902    0.000   17.785    0.000 module.py:1735(_wrapped_call_impl)
698733/140673    1.156    0.000   17.579    0.000 module.py:1743(_call_impl)
   139515    1.450    0.000   17.034    0.000 model.py:12(forward)
    46588    1.285    0.000   14.782    0.000 Agent.py:98(remember)
   558060    0.649    0.000   10.477    0.000 linear.py:124(forward)
    46589    1.013    0.000   10.187    0.000 Agent.py:132(get_action)
   558060    9.459    0.000    9.459    0.000 {built-in method torch._C._nn.linear}
      586    0.002    0.000    8.278    0.014 Agent.py:122(train_long_memory)
      586    0.058    0.000    8.276    0.014 model.py:95(train)
     1172    0.267    0.000    8.218    0.007 model.py:61(train_step)
    46619    0.295    0.000    5.962    0.000 main_screen.py:56(render_screen)
       89    0.002    0.000    5.681    0.064 __init__.py:1(<module>)
      574    0.001    0.000    5.010    0.009 Agent.py:163(check_training)
    46589    0.423    0.000    4.723    0.000 tetris.py:118(play)
    25/20    0.000    0.000    4.424    0.221 _ops.py:279(py_impl)
    46588    0.030    0.000    3.319    0.000 Agent.py:155(check_steps)
        3    0.000    0.000    2.942    0.981 _ops.py:147(py_functionalize_impl)
   495399    2.924    0.000    2.924    0.000 {method 'blit' of 'pygame.surface.Surface' objects}
   627556    0.958    0.000    2.835    0.000 tetris.py:87(move_down)
     1158    0.018    0.000    2.699    0.002 model.py:48(fit)
   418545    0.340    0.000    2.475    0.000 functional.py:1693(relu)
    46619    2.442    0.000    2.442    0.000 {built-in method pygame.display.update}
     1158    0.884    0.001    2.232    0.002 Agent.py:278(update_priority)
     1158    1.688    0.001    2.077    0.002 Agent.py:258(sample)
   418545    2.039    0.000    2.039    0.000 {built-in method torch.relu}
    90028    1.709    0.000    1.709    0.000 {built-in method numpy.array}
        3    0.000    0.000    1.485    0.495 _ops.py:251(__init__)
        1    0.000    0.000    1.457    1.457 triton_kernel_wrap.py:650(__init__)
"""