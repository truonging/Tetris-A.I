import pygame.time
import matplotlib

import matplotlib.pyplot as plt
from settings import *
from tetris import *
from Agent import *
from game_screen import *


class Main_Screen:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.FPS = [0,60]
        self.i = 0

        self.display_surface = pygame.display.set_mode((MAIN_WIDTH, MAIN_HEIGHT), pygame.SRCALPHA)
        self.grid_surface = pygame.Surface((MAIN_WIDTH, MAIN_HEIGHT), pygame.SRCALPHA)
        self.highlight_surface = pygame.Surface((MAIN_WIDTH, MAIN_HEIGHT), pygame.SRCALPHA)
        self.grid_surface.fill((0, 0, 0, 0))
        self.highlight_surface.fill((0,0,0,0))
        pygame.display.set_caption('Tetris')

        self.games = {i: Tetris(i) for i in range(1, POPULATION_SIZE + 1)}
        self.agents = {}
        self.screens = {i: Game_Screen(i) for i in range(1, POPULATION_SIZE+1)}
        self.old_states = {i: [0,0,0,0,0,0] for i in range(1, POPULATION_SIZE+1)}
        self.no_train = {i:0 for i in range(1, POPULATION_SIZE+1)}
        self.trained = {i:False for i in range(1, POPULATION_SIZE+1)}
        # self.scoreboards = {i: Scoreboard(self.display_surface, i) for i in range(1, POPULATION_SIZE+1)}
        self.main_scoreboard = Main_Scoreboard()

        for i in range(1, POPULATION_SIZE+1):
            self.draw_board_outline(self.grid_surface, i)
            # self.draw_board_gridlines(i)

        self.RENDER = True
        self.all_generations = []
        self.generation_avg_lines = self.generation_std_dev = 0
        self.avg_diversity = []
        self.generation = 1
        self.total_games_to_end = 0


    def reset(self):
        self.agents = {}
        self.old_states = {i: [0,0,0,0,0,0] for i in range(1, POPULATION_SIZE+1)}
        self.no_train = {i:0 for i in range(1, POPULATION_SIZE+1)}
        self.trained = {i:False for i in range(1, POPULATION_SIZE+1)}
        for i in range(1, POPULATION_SIZE+1):
            self.games[i].total_games = 1
            self.games[i].total_lines = 0
            self.games[i].hiscore = 0

    def render_screen(self, new_changes, data=None):
        # draw each new update
        for i, r, c, color_id in new_changes:
            self.screens[i].draw_block(self.display_surface, r, c, color_id)

        # get the current best performing genomes
        best_genomes = sorted(
            [(i, game.hiscore, game.total_lines/game.total_games, game.total_games)
             for i, game in self.games.items()],
            key=lambda x: x[1],
            reverse=True
        )

        # highlight the top 5
        #self.highlight_surface.fill((0, 0, 0, 0))
        #for j in range(5):
        #    self.draw_board_outline(self.highlight_surface, best_genomes[j][0], (255, 0, 0, 255), 3)

        # blit the updates and highlight onto main
        self.display_surface.blit(self.grid_surface, (0, 0), )
        self.display_surface.blit(self.highlight_surface, (0, 0), )

        # information display
        if data:
            generation, total_games_to_end, uniform_rate, mutate_rate, last_hiscore, last_avg_lines = data
            data = [self.display_surface, best_genomes, generation, total_games_to_end, uniform_rate, mutate_rate, self.agents[1].epsilon, self.agents[1].LR,last_hiscore,last_avg_lines]
            self.main_scoreboard.draw(data)

        # for i in range(1,POPULATION_SIZE+1):
        #     data = [self.games[i].total_games, self.games[i].lines, self.games[i].lines_removed]
        #     self.scoreboards[i].update(*data)
        #     self.scoreboards[i].draw()

        pygame.display.update()
        self.clock.tick(self.FPS[self.i])

    def draw_board_outline(self, surface, i, line_color='#a8a8a8', thickness=2):
        c, r = calc_i(i)
        # Calculate the position of the outline based on the grid position (c, r)
        outline_x = PADDING + (WINDOW_WIDTH * c)
        outline_y = PADDING + (WINDOW_HEIGHT * r)

        # Draw a rectangle around the real board (no padding for the outline)
        pygame.draw.rect(surface, line_color, (outline_x, outline_y, GAME_WIDTH, GAME_HEIGHT), thickness)

    def draw_board_gridlines(self, i):
        c, r = calc_i(i)
        # Calculate the top-left corner of the real board (inside the padding)
        grid_x = PADDING + (WINDOW_WIDTH * c)
        grid_y = PADDING + (WINDOW_HEIGHT * r)

        # Draw vertical lines
        for col in range(1, COLS):
            x = grid_x + (col * CELL_SIZE)
            pygame.draw.line(self.grid_surface, '##a8a8a8', (x, grid_y), (x, grid_y + GAME_HEIGHT))

        # Draw horizontal lines
        for row in range(1, ROWS):
            y = grid_y + (row * CELL_SIZE)
            pygame.draw.line(self.grid_surface, '##a8a8a8', (grid_x, y), (grid_x + GAME_WIDTH, y))

    def play_action(self, i):
        game, agent = self.games[i], self.agents[i]
        rotate_shapes = TETROMINOS_NUMPY[game.tetromino['key']]['shape']
        states = calc_all_states3(
            np.array(game.board, dtype=np.int64),
            game.tetromino['rotations'],
            rotate_shapes[0],
            rotate_shapes[90],
            rotate_shapes[180],
            rotate_shapes[270]
        )
        next_states = states
        if not next_states:
            return i, None, [], game, agent

        best_state = agent.get_action(next_states.keys())
        decode = next_states[best_state]
        x = decode // 4
        r = decode % 4
        best_action = (x, r)
        changes = game.play(best_action)
        return i, best_state, changes, game, agent

    def run2(self, population, total_games_to_end, generation, uniform_rate, mutate_rate, last_hiscore, last_avg_lines):
        self.display_surface.fill(GRAY)
        self.reset()
        self.agents = {i:population[i-1][1] for i in range(1,POPULATION_SIZE+1)}
        self.total_games_to_end = total_games_to_end
        self.generation = generation
        games_running = {i for i in range(1, POPULATION_SIZE+1)}
        key_pressed = False
        min_games_needed = int(total_games_to_end*0.4)

        # go until no more games running
        while games_running:
            for event in pygame.event.get():
                match event.type:
                    case pygame.QUIT:
                        pygame.quit()
                        exit()
                    case pygame.KEYDOWN if event.key == pygame.K_KP_MULTIPLY and not key_pressed:
                        self.RENDER = not self.RENDER
                        key_pressed = True
                    case pygame.KEYUP if event.key == pygame.K_KP_MULTIPLY:
                        key_pressed = False
                    case pygame.KEYDOWN if event.key == pygame.K_KP_DIVIDE and not key_pressed:
                        self.i = (self.i+1)%2
                        key_pressed = True
                    case pygame.KEYUP if event.key == pygame.K_KP_DIVIDE:
                        key_pressed = False

            # decide best action and play it
            results = [self.play_action(i) for i in games_running]
            new_changes = []

            for i, best_state, changes, game, agent in results:
                self.games[i] = game
                self.agents[i] = agent
                done,agent = self.games[i].finished, self.agents[i]

                # only add changes of games going on
                if not done:
                    new_changes += changes

                # gameover
                if done or not best_state:
                    # reached final game or performing badly
                    if (game.total_games == total_games_to_end) or (game.total_games==min_games_needed and self.check_performance(i)):
                        games_running.remove(i)
                        game.old_state = [0, 0, 0, 0, 0]
                        continue

                    # not final game, reset and keep going
                    game.total_games += 1

                    new_changes += game.clear_board()

                    game.reset()
                    #self.scoreboards[i].reset()

                    # less exploration, more exploiting
                    agent.decay_epsilon(game.total_games)
                    agent.calculate_lr(game.total_games)

                    # print(f'LR={agent.LR:.4f} |  Epsilon={agent.epsilon:.5f} at game={game.total_games}')

                    # train at least once a game
                    agent.check_training()

                    # no moves were possible, so nothing to calculate
                    if not best_state:
                        continue

                # train every 200 steps
                agent.check_steps()

                # calculate reward and add to memory
                reward = agent.calculate_rewards(best_state, done)
                agent.remember(game.old_state, best_state, reward, done)
                game.old_state = best_state

            if self.RENDER:
                data = [generation, total_games_to_end, uniform_rate, mutate_rate, last_hiscore, last_avg_lines]
                self.render_screen(new_changes, data)

        #self.plot()
        self.write_to_files()

        for i in range(1, POPULATION_SIZE+1):
            population[i-1][2] = float("{:.4f}".format(self.games[i].total_lines/total_games_to_end))
            # self.games[i].total_games = 1
            # self.games[i].total_lines = 0
            # self.games[i].hiscore = 0

        return population
  
    def check_performance(self, i):
        current_gen_avg = self.generation_avg_lines
        current_gen_std = self.generation_std_dev

        performance = self.games[i].total_lines / self.games[i].total_games

        # Confidence threshold (Mean - 2 * Std Dev)
        cutoff_threshold = current_gen_avg - 2 * current_gen_std
        if i%40==0:
            print(f'Agent {i}: Avg={performance:.2f}, Gen Avg={current_gen_avg:.2f}, Std={current_gen_std:.2f}, Threshold={cutoff_threshold:.2f}')
        if performance < cutoff_threshold:
            print(f"Agent {i} cut off early. Avg = {performance:.2f}, Threshold = {cutoff_threshold:.2f}")
            return True
        return False

    def write_to_files(self):
        avg_lines_per_ai = [self.games[i].total_lines/self.games[i].total_games for i in range(1,POPULATION_SIZE+1)]
        genomes = [list(self.agents[i].weight.values()) for i in range(1,POPULATION_SIZE+1)]
        differences = []
        for g1, g2 in combinations(genomes, 2):
            differences.append(np.sum(np.abs(np.array(g1) - np.array(g2))))

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Create the "data" folder in the same directory as the script if it doesn't exist
        data_folder = os.path.join(script_dir, 'data')
        os.makedirs(data_folder, exist_ok=True)

        # Define file paths
        file_paths = {
            "all_generations": os.path.join(data_folder, "all_generations.txt"),
            "generation_mean_line": os.path.join(data_folder, "generation_mean_line.txt"),
            "generation_std_dev": os.path.join(data_folder, "generation_std_dev.txt"),
            "diversity": os.path.join(data_folder, "diversity.txt")
        }

        # Data to write to each file
        avg_lines = np.mean(avg_lines_per_ai)
        avg_std_dev = np.std(avg_lines_per_ai)
        data_to_write = {
            "all_generations": avg_lines_per_ai,
            "generation_mean_line": avg_lines,
            "generation_std_dev": avg_std_dev,
            "diversity": np.mean(differences)
        }
        self.generation_avg_lines = avg_lines
        self.generation_std_dev = avg_std_dev

        # Write to each file
        for file_name, file_path in file_paths.items():
            with open(file_path, "a") as file:  # "a" mode appends data to the file
                file.write(str(data_to_write[file_name]) + "\n")

        print("Data successfully written to files.")


def run_game():
    t = Main_Screen()
    return t.run()


import cProfile
import pstats

if __name__=='__main__':
    pass
    # main_screen = Main_Screen()
    # main_screen.run()
    # cProfile.run('run_game()', 'profile_output.prof')
    # p = pstats.Stats('profile_output.prof')
    # p.strip_dirs().sort_stats('cumulative').print_stats(lambda x: x >= 1)