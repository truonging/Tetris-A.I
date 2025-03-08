from settings import *
from os.path import join

class Scoreboard:
    def __init__(self):
        self.surface = pygame.Surface((SIDEBAR_WIDTH, GAME_HEIGHT * 1))
        self.rect = self.surface.get_rect(topright=(WINDOW_WIDTH - PADDING, WINDOW_HEIGHT - GAME_HEIGHT - PADDING))
        self.display_surface = pygame.display.get_surface()

        self.font = pygame.font.Font(join('assets', 'graphics', 'Russo_One.ttf'), 15)

        self.increment_height = self.surface.get_height() / 14

        self.level = 1
        self.score = 0
        self.lines = 0
        self.games = 1
        self.total_lines = 0
        self.hiscore = 0
        self.line_clears = {1: 0, 2: 0, 3: 0, 4: 0}
        self.total_line_clears = {1: 0, 2: 0, 3: 0, 4: 0}
        self.epsilon = 0.3

    def update_from(self, game, epsilon):
        self.level = game.level
        self.score = game.score
        self.lines = game.lines
        self.total_lines += game.lines_removed
        self.hiscore = max(self.hiscore, self.score)

        # Update line clear counts
        if game.lines_removed in self.line_clears:
            self.line_clears[game.lines_removed] += 1
            self.total_line_clears[game.lines_removed] += 1

        # (randomness percentage)
        self.epsilon = epsilon * 100

    def reset(self):
        self.level = 1
        self.score = 0
        self.lines = 0
        self.games = 1
        self.line_clears = {1: 0, 2: 0, 3: 0, 4: 0}
        self.epsilon = 0

    def display_text(self, pos, text):
        text_surface = self.font.render(text, True, 'white')
        text_rect = text_surface.get_rect(center=pos)
        self.surface.blit(text_surface, text_rect)

    def run(self, game, game_num, epsilon):
        self.surface.fill(GRAY)
        self.update_from(game, epsilon)

        # Extra padding for spacing
        padding_offset = 15

        # Main stats to display
        stats = [
            ('Game', game_num),
            ('Level', self.lines//10),
            ('Hiscore', f"{self.hiscore:,}"),
            ('Score', f"{self.score:,}"),
            ('Lines', self.lines),
            ('Avg Lines', f"{self.total_lines / game_num:.2f}"),
            ('Line Clears', '')  # Header for line clear stats
        ]

        # Render the main stats
        for i, (label, value) in enumerate(stats):
            x = self.surface.get_width() / 2
            y = padding_offset + (self.increment_height / 10) + i * self.increment_height  # Normal spacing
            self.display_text((x, y), f'{label:<8}: {value}')

        # Define column headers for the Line Clears table
        line_clear_header_y = y + self.increment_height  # Start below "Line Clears" header
        header_x_positions = [self.surface.get_width() * 0.10,  # Left (Clear Type)
                              self.surface.get_width() * 0.35,  # Middle (Current)
                              self.surface.get_width() * 0.60,  # Right (Total)
                              self.surface.get_width() * 0.80]  # Percentage

        # Render the headers
        headers = ["Type", "Current", "Total"]
        for i, header in enumerate(headers):
            self.display_text((header_x_positions[i], line_clear_header_y), header)

        # Render line clear stats (aligned in 3 columns)
        total_all_line_clears = sum(self.total_line_clears.values())
        line_clear_start_y = line_clear_header_y + self.increment_height / 2
        line_clear_spacing = self.increment_height / 1.5  # Tighter spacing

        for i in range(1, 5):  # Loop through 1-line to 4-line clears
            y = line_clear_start_y + (i - 1) * line_clear_spacing
            line_clear_count = self.total_line_clears[i]
            percentage = (line_clear_count / total_all_line_clears * 100) if total_all_line_clears > 0 else 0
            self.display_text((header_x_positions[0], y), f"{i}")  # Type (1,2,3,4)
            self.display_text((header_x_positions[1], y), f"{self.line_clears[i]:>3}")  # Current game
            self.display_text((header_x_positions[2], y), f"{self.total_line_clears[i]:>3}")  # Total clears
            self.display_text((header_x_positions[3], y), f"{percentage:>5.1f}%")  # Overall percentage

        # Render "Random %" under Line Clears
        random_y = y + self.increment_height  # Position it below the line clear table
        self.display_text((self.surface.get_width() / 2, random_y), f"Random Move: {self.epsilon:.2f}%")

        # Display the scoreboard
        self.display_surface.blit(self.surface, self.rect)
        pygame.draw.rect(self.display_surface, LINE_COLOR, self.rect, 2, 2)