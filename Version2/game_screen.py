from settings import *


class Game_Screen:
    """
    acts as the container for board and scoreboard
    """
    def __init__(self, i):
        c, r = calc_i(i)

        # Calculate the top-left corner for this board on main_display
        board_x = PADDING + (WINDOW_WIDTH * c)
        board_y = PADDING + (WINDOW_HEIGHT * r)

        # Display
        # self.rect = pygame.Rect(WINDOW_WIDTH * c, WINDOW_HEIGHT * r, WINDOW_WIDTH, WINDOW_HEIGHT)
        # self.display_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        # self.rect = pygame.Rect(WINDOW_WIDTH * c + PADDING, WINDOW_HEIGHT * r + PADDING, WINDOW_WIDTH, WINDOW_HEIGHT)
        # self.board = Board(self.display_surface, i, dirty_rect)
        #self.scoreboard = Scoreboard(self.display_surface, i)
        #self.display_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        # self.rect = self.display_surface.get_rect(topleft=(0,0))
        #self.rect = self.display_surface.get_rect(topleft=(PADDING, PADDING))

        # Static container for game screen border
        # self.container_border = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        # pygame.draw.rect(self.container_border, LINE_COLOR, (0, 0, GAME_WIDTH, GAME_HEIGHT), 2, 2)
        # self.container_border.set_alpha(180)
        # self.display_surface.blit(self.container_border, self.rect.topleft)
        # self.display_surface.blit(self.container_border, (0,0))

        # Precompute cell rectangles for faster blitting
        #self.cells = [[pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE) for c in range(COLS)] for r in range(ROWS)]
        self.cells = [[pygame.Rect(
            board_x + (c * CELL_SIZE),  # Absolute x position
            board_y + (r * CELL_SIZE),  # Absolute y position
            CELL_SIZE,
            CELL_SIZE
        ) for c in range(COLS)] for r in range(ROWS)]

        self.block_surfaces = {BLOCK_COLORS[color]: pygame.Surface((CELL_SIZE, CELL_SIZE)) for color in BLOCK_COLORS}
        for color, surf in self.block_surfaces.items():
            surf.fill(color)

        # self.clear()

    #def clear(self):
    #    self.display_surface.fill(GRAY)

    def draw_block(self, main_display, r, c, color_id):
        main_display.blit(self.block_surfaces[BLOCK_COLORS[color_id]], self.cells[r][c].topleft)

    def draw(self):
        #self.board.draw(board)

        #self.scoreboard.draw()

        # self.display_surface.blit(self.container_border, self.rect.topleft)
        # self.display_surface.blit(self.container_border, (0,0))
        # pygame.draw.rect(self.display_surface, LINE_COLOR, self.board.rect, 2, 2)
        #pygame.draw.rect(self.display_surface, LINE_COLOR, self.scoreboard.rect, 2, 2)
        pass

class Board:
    def __init__(self, display_surface, i, dirty_rect):
        c, r = calc_i(i)
        self.i = i

        # Display
        self.display_surface = display_surface
        self.surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT), pygame.SRCALPHA)
        self.rect = self.surface.get_rect(topleft=(PADDING, PADDING))
        #self.rect = self.surface.get_rect(topleft=(PADDING + (WINDOW_WIDTH * c), PADDING + (WINDOW_HEIGHT * r)))

        # Cached grid lines
        self.line_surface = pygame.Surface(self.surface.get_size(), pygame.SRCALPHA)
        self.line_surface.fill((0, 0, 0, 0))  # Transparent background
        self.draw_grid()
        self.line_surface.set_alpha(120)

        self.dirty_rect = dirty_rect
        self.start = False

        # Precompute cell rectangles for faster blitting
        self.cells = [[pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE) for c in range(COLS)] for r in range(ROWS)]
        self.old = [[0]*COLS for _ in range(ROWS)]
        self.prev_board = [row[:] for row in self.old]

        self.block_surfaces = {BLOCK_COLORS[color]: pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA) for color in BLOCK_COLORS}
        for color, surf in self.block_surfaces.items():
            surf.fill(color)

    def reset(self):
        self.prev_board = [row[:] for row in self.old]
        self.surface.fill(GRAY)

    def draw_grid(self):
        """ Draws the grid lines once and caches them. """
        r = self.surface.get_height()
        for col in range(1, COLS):
            c = col * CELL_SIZE
            pygame.draw.line(self.line_surface, LINE_COLOR, (c, 0), (c, r), 1)
        c = self.surface.get_width()
        for row in range(1, ROWS):
            r = row * CELL_SIZE
            pygame.draw.line(self.line_surface, LINE_COLOR, (0, r), (c, r), 1)

    def draw_board(self, board):
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c]:
                    pygame.draw.rect(self.surface, BLOCK_COLORS[board[r][c]], self.cells[r][c])

    def draw_piece(self, blocks):
        for block in blocks:
            r,c = block.pos
            pygame.draw.rect(self.surface, BLOCK_COLORS[board[r][c]], self.cells[r][c])
            self.prev_board[r][c] = board[r][c]

    def draw_updated_board(self, board):
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c]!=self.prev_board[r][c]:
                    pygame.draw.rect(self.surface, BLOCK_COLORS[board[r][c]], self.cells[r][c])

    def draw(self, board):
        # # if cleared lines, redo board
        # if lines_cleared:
        #     self.draw_updated_board(board)
        # # else update piece only
        # self.draw_piece(blocks)
        # Only clear changed rows
        changed_rows = set()
        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] != self.prev_board[r][c]:
                    changed_rows.add(r)
                    # Clear the changed cell
                    #pygame.draw.rect(self.surface, GRAY, self.cells[r][c])
                    self.surface.blit(self.block_surfaces[GRAY], self.cells[r][c])

        # Draw only the changed rows
        for r in changed_rows:
            for c in range(COLS):
                if board[r][c]:
                    self.surface.blit(self.block_surfaces[BLOCK_COLORS[board[r][c]]], self.cells[r][c])
        self.prev_board = [row[:] for row in board]

        # self.surface.fill(GRAY)
        # self.draw_board(board)

        # Update current dirty then draw onto gameboard
        # self.dirty_rect.update()
        # self.dirty_rect.draw(self.surface)

        # Blit cached grid lines
        self.surface.blit(self.line_surface, (0, 0))

        # Blit to display surface
        self.display_surface.blit(self.surface, self.rect)


# class Scoreboard:
#     """
#     container for the sidebar scoreboard
#     """
#     def __init__(self, display_surface, i):
#         c, r = calc_i(i)
#         self.surface = pygame.Surface((SIDEBAR_WIDTH,GAME_HEIGHT*SCORE_HEIGHT_FRACTION), pygame.SRCALPHA)
#         #self.rect = self.surface.get_rect(topright=(WINDOW_WIDTH-PADDING+(WINDOW_WIDTH*c),PADDING+(WINDOW_HEIGHT*r)))
#         self.rect = self.surface.get_rect(topright=(WINDOW_WIDTH - PADDING, PADDING))
#         self.display_surface = display_surface
#         self.increment_height = self.surface.get_height() / 3
#         self.font = pygame.font.Font(join('assets','graphics','Russo_One.ttf'),int(15))
#
#         self.lines = 0
#         self.games = 1
#         self.total_lines = 0
#
#         # Cache text surfaces (Tip #4)
#         self.prev_texts = ["", "", ""]  # To store previous text states
#         self.text_surfaces = [None, None, None]  # To store pre-rendered text surfaces
#
#     def update(self, game_num, lines, lines_removed):
#         self.games = game_num
#         self.lines = lines
#         self.total_lines += lines_removed
#
#     def display_text(self):
#         texts = [('Game', self.games), ("Lines", self.lines), ('Avg', f"{self.total_lines/self.games:.2f}")]
#         for i, (label, value) in enumerate(texts):
#             new_text = f'{label}: {value}'
#             if self.prev_texts[i] != new_text:  # Only update if text changed
#                 self.prev_texts[i] = new_text
#                 self.text_surfaces[i] = self.font.render(new_text, True, 'white')
#
#             text_rect = self.text_surfaces[i].get_rect(center=(self.surface.get_width() / 2,
#                                                               self.increment_height / 3 + i * self.increment_height))
#             self.display_surface.blit(self.text_surfaces[i], text_rect)
#
#     def draw(self):
#         self.surface.fill(GRAY)
#         self.display_text()
#
#         self.display_surface.blit(self.surface, self.rect)
#         #pygame.draw.rect(self.display_surface, LINE_COLOR, self.rect, 2, 2)
#

class Scoreboard:
    """
    Container for the sidebar scoreboard, directly drawing to the main display.
    """

    def __init__(self, display_surface, i):
        c, r = calc_i(i)  # Calculate the grid position based on the index `i`

        # Calculate the top-left position for the scoreboard on the main display
        self.x =  -PADDING + (WINDOW_WIDTH * c) + (WINDOW_WIDTH - SIDEBAR_WIDTH)  # Right side of the board
        self.y = PADDING + (WINDOW_HEIGHT * r)  # Align with the grid's row

        self.display_surface = display_surface
        self.increment_height = GAME_HEIGHT * SCORE_HEIGHT_FRACTION / 4
        self.font = pygame.font.Font(join('assets', 'graphics', 'Russo_One.ttf'), int(CELL_SIZE*.85))

        self.lines = 0
        self.games = 1
        self.total_lines = 0
        self.hiscore = 0
        self.score = 0

        # Cache text surfaces
        self.prev_texts = ["", "", "", "", ""]
        self.text_surfaces = [None, None, None, None, None]

    def reset(self):
        self.lines = 0
        self.score = 0

    def update(self, game_num, lines, lines_removed):
        self.games = game_num
        self.lines = lines
        self.total_lines += lines_removed
        self.score += SCORE_DATA[lines_removed] * (self.lines//10+1)
        self.hiscore = max(self.hiscore, self.score)

    def display_text(self):
        texts = [
            ('Game', self.games),
            ('Hiscore', f"{self.hiscore}"),
            ("Score", self.score),
            ("Lines", self.lines),
            ('Avg', f"{self.total_lines/self.games :.2f}")
        ]
        for i, (label, value) in enumerate(texts):
            new_text = f'{label}: {value}'
            if self.prev_texts[i] != new_text:  # Only update if text changed
                self.prev_texts[i] = new_text
                self.text_surfaces[i] = self.font.render(new_text, True, 'white')

            # Calculate the position to center the text on the scoreboard
            text_rect = self.text_surfaces[i].get_rect(center=(self.x + SIDEBAR_WIDTH / 2,
                                                               self.y + self.increment_height / 4 + i * self.increment_height))
            self.display_surface.blit(self.text_surfaces[i], text_rect)

    def draw(self):
        scoreboard_area_rect = pygame.Rect(self.x, self.y, SIDEBAR_WIDTH, GAME_HEIGHT)
        self.display_surface.fill(GRAY, scoreboard_area_rect)
        # Draw the scoreboard text directly onto the main display
        self.display_text()

class Main_Scoreboard:
    def __init__(self):
        x = MAIN_WIDTH - MAIN_SIDEBAR_WIDTH
        y = 0
        self.rect = pygame.Rect(x, y, MAIN_SIDEBAR_WIDTH, MAIN_HEIGHT)

        # Create surface for the scoreboard
        self.surface = pygame.Surface((MAIN_SIDEBAR_WIDTH, MAIN_HEIGHT))
        self.surface.set_alpha(180)  # Slight transparency for better visibility

        # Font setup
        self.font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self.small_font = pygame.font.Font(pygame.font.get_default_font(), 14)

        # Padding and spacing
        self.padding = 8
        self.line_spacing = 20

    def draw(self, data):
        # Clear the surface
        self.surface.fill((30, 30, 30))
        main_display,best_genomes,generation,total_games,uniform_pct,mutate_pct,epsilon,learning_rate,last_hiscore,last_avg_lines = data
        # Display the statistics at the top
        mn = min(best_genomes,key=lambda x:x[3])[3]
        #mx = max(best_genomes,key=lambda x:x[3])[3]
        mx = max(p[3] if p[3]!=250 else 0 for p in best_genomes)
        stats = [
            f"Generation: {generation} out of 100",
            f"Population: {POPULATION_SIZE}",
            f"Total Games: {total_games}",
            f"Lowest Game:  {mn}",
            f"Highest Game: {mx}",
            # f"Uniform: {uniform_pct[0]:.2f}",
            # f"Alpha: {uniform_pct[1]:.2f}",
            f"Mutate: {mutate_pct:.2f}%",
            #f"Epsilon: {epsilon:.4f}",
            #f"LR: {learning_rate:.4f}"
        ]

        y_offset = self.padding
        for stat in stats:
            text_surface = self.font.render(stat, True, (255, 255, 255))
            self.surface.blit(text_surface, (self.padding, y_offset))
            y_offset += self.line_spacing

        # --- Draw High Score Leaderboard ---
        leaderboard_title = self.font.render("Leaderboard (Hiscore)", True, (255, 215, 0))
        self.surface.blit(leaderboard_title, (self.padding, y_offset))
        y_offset += self.line_spacing

        text = f"Last Gen Hiscore: {last_hiscore}"
        text_surface = self.small_font.render(text, True , (200, 200, 200))
        self.surface.blit(text_surface, (self.padding, y_offset))
        y_offset += self.line_spacing

        # Only show top 15 by hiscore
        for rank, (i, score, avg_lines, _) in enumerate(best_genomes[:15], start=1):
            text = f"{rank}. Hiscore: {score}"
            text_surface = self.small_font.render(text, True, (200, 200, 200))
            self.surface.blit(text_surface, (self.padding, y_offset))
            y_offset += self.line_spacing

        # --- Sort and Draw Average Lines Leaderboard ---
        y_offset += self.line_spacing  # Extra spacing before the second leaderboard
        avg_lines_title = self.font.render("Leaderboard (Avg Lines)", True, (255, 215, 0))
        self.surface.blit(avg_lines_title, (self.padding, y_offset))
        y_offset += self.line_spacing

        text = f"Last Gen Avg Lines: {last_avg_lines:.2f}"
        text_surface = self.small_font.render(text, True , (200, 200, 200))
        self.surface.blit(text_surface, (self.padding, y_offset))
        y_offset += self.line_spacing

        # Sort by average lines in descending order and take top 15
        best_by_avg_lines = sorted(best_genomes, key=lambda x: x[2], reverse=True)[:15]

        for rank, (i, score, avg_lines, _) in enumerate(best_by_avg_lines, start=1):
            text = f"{rank}. Avg Lines: {avg_lines:.2f}"
            text_surface = self.small_font.render(text, True, (200, 200, 200))
            self.surface.blit(text_surface, (self.padding, y_offset))
            y_offset += self.line_spacing

        # Blit the scoreboard to the main display
        main_display.blit(self.surface, self.rect.topleft)