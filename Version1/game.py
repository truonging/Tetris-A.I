from random import choice,random

from settings import *
from timer import Timer
from njit_startup import *

class Game:
    def __init__(self, player=False):
        self.player = player
        # display
        self.surface = pygame.Surface((GAME_WIDTH,GAME_HEIGHT))
        self.display_surface = pygame.display.get_surface()
        self.rect = self.surface.get_rect(topleft=(PADDING,PADDING))
        self.sprites = pygame.sprite.Group()
        self.line_surface = self.surface.copy()
        self.line_surface.fill((0,255,0))
        self.line_surface.set_colorkey((0,255,0))
        self.line_surface.set_alpha(120)

        # # timing
        self.normal_speed = UPDATE_START_SPEED
        self.fast_speed = self.normal_speed * 0.5
        self.pressed_down = False
        self.timers = {
            'vertical move': Timer(self.normal_speed,True,self.move_down),
            'horizontal move': Timer(MOVE_WAIT_TIME),
            'rotate': Timer(ROTATE_WAIT_TIME),
            'hard drop': Timer(HARD_DROP_WAIT_TIME)
        }
        if self.player:
            self.timers['vertical move'].activate()

        # variables
        self.level = 1
        self.score = 0
        self.lines = 0
        self.dirs = 0
        self.lines_removed = 0
        self.finished = False
        self.collision = False

        # board
        self.board = np.zeros((ROWS, COLUMNS), dtype=int)
        self.board_blocks = {}
        self.block_id = 2

        # tetromino
        self.previews = [TETROMINOS_KEYS[int(random()*KEY_LEN)] for _ in range(2)]
        self.tetromino = Tetromino(
            choice(TETROMINOS_KEYS),
            self.sprites,
            self.board,
            self.set_gameover,
            self.set_collision,
            self.update_board_blocks,
            self.player
        )

    def update_board_blocks(self, blocks, update_game, board):
        for block in blocks:
            board[int(block.pos.y)][int(block.pos.x)] = self.block_id if update_game else 1
            if update_game:
                self.board_blocks[self.block_id] = block
                self.block_id += 1

    def _display_game(self):
        self.surface.fill(GRAY)
        self.sprites.draw(self.surface)
        y = self.surface.get_height()

        # draw grid
        for col in range(1,COLUMNS):
            x = col*CELL_SIZE
            pygame.draw.line(self.line_surface,LINE_COLOR,(x,0),(x,y),1)
        x = self.surface.get_width()
        for row in range(1,ROWS):
            y = row*CELL_SIZE
            pygame.draw.line(self.line_surface,LINE_COLOR,(0,y),(x,y),1)

        self.surface.blit(self.line_surface,(0,0))
        self.display_surface.blit(self.surface, (PADDING,PADDING))
        pygame.draw.rect(self.display_surface,LINE_COLOR,self.rect,2,2)

    def _create_fake_board(self):
        return (self.board != 0).astype(int)

    def _input(self):
        if self.player:
            keys = pygame.key.get_pressed()
            # rotate
            if not self.timers['rotate'].active and keys[pygame.K_UP]:
                self.tetromino.rotate()
                self.timers['rotate'].activate()

            # hard drop
            if not self.timers['hard drop'].active:
                if keys[pygame.K_a]:
                    self.tetromino.hard_drop(self.board)
                    self.timers['hard drop'].activate()

            # left or right
            if not self.timers['horizontal move'].active:
                if keys[pygame.K_LEFT]:
                    self.tetromino.move_horizontal(-1)
                    self.timers['horizontal move'].activate()
                elif keys[pygame.K_RIGHT]:
                    self.tetromino.move_horizontal(1)
                    self.timers['horizontal move'].activate()

            # press down button
            if not self.pressed_down and keys[pygame.K_DOWN]:
                self.pressed_down = True
                self.timers['vertical move'].duration = self.fast_speed

            # release down button
            if self.pressed_down and not keys[pygame.K_DOWN]:
                self.pressed_down = False
                self.timers['vertical move'].duration = self.normal_speed

        if self.dirs==-1:
            self.tetromino.move_horizontal(-1)
        elif self.dirs==1:
            self.tetromino.move_horizontal(1)
        elif self.dirs==2:
            self.move_down()
        elif self.dirs==3:
            self.tetromino.rotate()
        elif self.dirs==4:
            self.tetromino.hard_drop()

        if self.collision:
            self._update_board()

    def _update_board(self):
        full_rows = self.board[np.all(self.board > 0, axis=1)]
        self._kill_tetromino(full_rows)
        self.board, lines_cleared = self._delete_lines(self.board, full_rows)
        self._update_tetro_pos()
        self.calc_score(lines_cleared)
        self.create_tetromino()

    def _delete_lines(self, board, full_rows):
        lines_cleared = len(full_rows)
        if len(full_rows)>0:
            rows = np.where(np.all(board > 0, axis=1))[0]
            board = np.delete(board, rows, axis=0)
            new_rows = np.zeros((lines_cleared, COLUMNS), dtype=int)
            board = np.vstack((new_rows, board))
        return board, lines_cleared

    def _kill_tetromino(self, rows):
        for row in rows:
            for block in row:
                if block > 1:
                    self.board_blocks[block].kill()
                    del self.board_blocks[block]

    def _update_tetro_pos(self):
        for r in range(ROWS):
            for c in range(COLUMNS):
                if self.board[r][c] > 1:
                    block = self.board_blocks[self.board[r][c]]
                    block.pos.y += r - block.pos.y

    def set_collision(self):
        self.collision = True

    def set_gameover(self,boolean=False):
        self.finished = boolean

    def reset_turn_info(self):
        self.lines_removed = 0
        self.collision = False
        if self.player:
            for timer in self.timers.values():
                timer.update()

    def create_tetromino(self):
        self.tetromino = Tetromino(
            self.previews.pop(0),
            self.sprites,
            self.board,
            self.set_gameover,
            self.set_collision,
            self.update_board_blocks,
            self.player
        )
        self.previews += [choice(TETROMINOS_KEYS)]

    def calc_score(self, lines):
        self.score += SCORE_DATA[lines] * (self.lines//10+1)
        self.lines += lines
        self.lines_removed = lines

        # if self.lines / 10 > self.level:
            #self.level += 1
            # self.normal_speed *= 1
            # self.fast_speed = self.normal_speed * 0.3

    def move_down(self):
        self.tetromino.move_down(self.board)

    def calc_all_states(self):
        piece = self.tetromino
        states = {}
        for r in range(TETROMINOS[piece.shape]['rotations']):
            # figure out the farthest left/right the pivot can be
            mn = int(min(block.pos.x for block in piece.blocks))
            piv_x = mn
            left_bound = piv_x - mn
            right_bound = piv_x + 10 - int(max(block.pos.x for block in piece.blocks))

            for x in range(left_bound, right_bound):
                # adjust piece to new pivot
                if not piece.move_to_x(x):
                    continue
                board = self._create_fake_board()
                # hard drop and record board
                piece.hard_drop(board,False)

                # update if all pieces are valid
                if all(0<=block.pos.y<ROWS for block in piece.blocks):
                    full_rows = board[np.all(board > 0, axis=1)]
                    board, lines_removed = self._delete_lines(board, full_rows)
                    x_pivot = min(block.pos.x for block in piece.blocks)
                    states[(x_pivot,r)] = self.get_state(board,lines_removed)

                # after dropping, move back up
                piece.move_to_y(0)

            # after placing all, move back to middle and rotate
            piece.move_to_x(4)
            piece.rotate()

        return states

    def get_state(self, board=None, lines_removed=0):
        if board is None:
            board = self.board

        y_pos = max(block.pos.y for block in self.tetromino.blocks)
        cols, total_heights, bumpiness = get_states_fast(board)
        pillar = any(cols[i-1]-cols[i]>=3 and cols[i+1]-cols[i]>=3 for i in range(1, len(cols)-1)) or cols[1]-cols[0]>=3 or cols[-2]-cols[-1]>=3
        holes = np.sum((board == 0) & (np.cumsum(board != 0, axis=0) > 0))

        state = [total_heights, bumpiness, lines_removed, holes, y_pos, pillar]

        return state

    def run(self):
        self.reset_turn_info()
        if RENDER:
            self._display_game()

        # update
        self._input()

        # display
        if RENDER:
            self.sprites.update()


class Tetromino:
    def __init__(self, shape, group, board, set_gameover, set_collision, update_board_blocks, player):
        self.board = board
        self.shape = shape
        self.set_gameover = set_gameover
        self.set_collision = set_collision
        self.update_board_blocks = update_board_blocks
        self.block_positions = TETROMINOS[shape]['shape']
        self.color = TETROMINOS[shape]['color']
        self.blocks = [Block(group,pos,self.color) for pos in self.block_positions]
        self.player = player

        self._check_spawn()

    def _check_spawn(self):
        # if any block exist on a spawn block
        if any(self.board[int(block.pos.y)][int(block.pos.x)] if block.pos.y>=0 else False for block in self.blocks):
            self.set_gameover(True)
            return

    def move_to_x(self, x):
        # assuming left most block is pivot, move pivot to x
        diff = x - min(block.pos.x for block in self.blocks)
        for block in self.blocks:
            block.pos.x += diff

        return not any(block.pos.x>9 or block.pos.x<0 or (block.pos.y>=0 and self.board[int(block.pos.y)][int(block.pos.x)]>0) for block in self.blocks)

    def move_to_y(self,y):
        diff = int(self.blocks[0].pos.y) - y
        for block in self.blocks:
            block.pos.y -= diff

    def rotate(self):
        if self.shape == 'O': return
        pivot_pos = self.blocks[0].pos
        # collision check (probably don't need to check for RL)
        if self.player:
            new_blocks = [calc_pos(block,pivot_pos) for block in self.blocks]
            for pos in new_blocks:
                r,c = int(pos.y),int(pos.x)
                if not r<ROWS or not 0<=c<COLUMNS or (r>=0 and self.board[r][c]):
                    return False
        for i in range(len(self.blocks)):
            self.blocks[i].pos = calc_pos(self.blocks[i], pivot_pos)

        return True

    def hard_drop(self,board=None,update_game=True):
        while True:
            if self.move_down(board,update_game):
                return

    def move_down(self,board,update_game=True):
        if update_game:
            board = self.board
        collision = lambda x,y: (not y+1 <= ROWS-1) or (y>=0 and board[y+1][x]>0)
        # a piece touched the bottom of grid or another piece
        if any(collision(int(block.pos.x),int(block.pos.y)) for block in self.blocks):
            # record current tetromino
            self.update_board_blocks(self.blocks, update_game, board)

            if update_game:
                self.set_collision()
                # a piece exist out of bounds
                if any(block.pos.y < 0 for block in self.blocks):
                    self.set_gameover(True)

            return True

        # move blocks down one
        for block in self.blocks:
            block.pos.y += 1

        return False

    def move_horizontal(self, amount):
        collisionH = lambda x,y: not 0<=x+amount<COLUMNS or (y>=0 and self.board[y][x+amount])
        if any(collisionH(int(block.pos.x),int(block.pos.y)) for block in self.blocks):
            return

        for block in self.blocks:
            block.pos.x += amount

    def set_outline(self):
        for block in self.blocks:
            block.draw_outline(self.blocks)


class Block(pygame.sprite.Sprite):
    def __init__(self,group,pos,color):
        super().__init__(group)
        self.group = group
        self.color = color

        # display
        self.image = pygame.Surface((CELL_SIZE,CELL_SIZE))
        self.image.fill(self.color)
        self.pos = pygame.Vector2(pos) + BLOCK_OFFSET
        self.rect = self.image.get_rect(topleft=self.pos*CELL_SIZE)

    def update(self):
        self.rect.topleft = self.pos*CELL_SIZE

    def __repr__(self):
        return '1'

    def draw_outline(self, blocks):
        outline_color = '#FFFFFF'
        pygame.draw.rect(self.image, outline_color, (0, 0, CELL_SIZE, CELL_SIZE), width=2)