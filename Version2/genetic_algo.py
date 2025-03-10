import numpy as np
from settings import *
from random import sample, uniform, random
from Agent import *
from main_screen import *
import ast
from collections import deque

WEIGHT = {
    'game_over': (-300,-50),
    'survival_instinct': (20, 200),
    "total_height": (-20, 0),
    "lines_removed": (5, 50),
    "holes": (-40, 0),
    "bumpiness": (-40, 0),
    "pillar": (-40, 0),
    'y_pos_reward':(50,300),
    'y_pos_punish': (50,300)
}
"""
100 generation
random population of 200
50/50 move on       priority: lines -> survival
tournment selection for offsprings
uniform 100% -> 0%     (10% decrement every 10) generation
alpha   0% -> 100%     (10% increment every 10) generation
mutate  20% -> 5%      (1.5% decrement every 10) generation
games   500      (50 increment every 50) generation
pick    5 -> 15         (1 increment every 10) generation

epsilon 0.3 -> 0.01       (0.3 -> 0.01 in exactly games) games
decay rate based on total games
LR      0.01 -> 0.001  (decrement every game til 500) games
gamma   0.97
"""

class Genetic_Algo:
    def __init__(self):
        self.main_screen = Main_Screen()
        self.generation_passed = 1
        self.TOTAL_GAMES = TOTAL_GAMES

    def create_population(self, n=POPULATION_SIZE):
        # return [
        #     [{
        #         'game_over': np.random.uniform(WEIGHT['game_over'][0], WEIGHT['game_over'][1]),
        #         'survival_instinct': np.random.uniform(WEIGHT['survival_instinct'][0], WEIGHT['survival_instinct'][1]),
        #         "total_height": np.random.uniform(WEIGHT['total_height'][0], WEIGHT['total_height'][1]),
        #         "lines_removed": np.random.uniform(WEIGHT['lines_removed'][0], WEIGHT['lines_removed'][1]),
        #         "holes": np.random.uniform(WEIGHT['holes'][0], WEIGHT['holes'][1]),
        #         "bumpiness": np.random.uniform(WEIGHT['bumpiness'][0], WEIGHT['bumpiness'][1]),
        #         "pillar": np.random.uniform(WEIGHT['pillar'][0], WEIGHT['pillar'][1]),
        #         'y_pos_reward': np.random.uniform(WEIGHT['y_pos_reward'][0], WEIGHT['y_pos_reward'][1]),
        #         #'y_pos_punish': np.random.uniform(WEIGHT['y_pos_punish'][0], WEIGHT['y_pos_punish'][1])
        #     }, Agent(), 0]
        #     for _ in range(n)
        # ]
        population = []
        for _ in range(n-1):
            genome = {
                'game_over': np.random.uniform(WEIGHT['game_over'][0], WEIGHT['game_over'][1]),
                'survival_instinct': np.random.uniform(WEIGHT['survival_instinct'][0], WEIGHT['survival_instinct'][1]),
                "total_height": np.random.uniform(WEIGHT['total_height'][0], WEIGHT['total_height'][1]),
                "lines_removed": np.random.uniform(WEIGHT['lines_removed'][0], WEIGHT['lines_removed'][1]),
                "holes": np.random.uniform(WEIGHT['holes'][0], WEIGHT['holes'][1]),
                "bumpiness": np.random.uniform(WEIGHT['bumpiness'][0], WEIGHT['bumpiness'][1]),
                "pillar": np.random.uniform(WEIGHT['pillar'][0], WEIGHT['pillar'][1]),
                "y_pos_reward": np.random.uniform(WEIGHT['y_pos_reward'][0], WEIGHT['y_pos_reward'][1]),
                "y_pos_punish": np.random.uniform(WEIGHT['y_pos_punish'][0], WEIGHT['y_pos_punish'][1]),
            }
            population.append([genome,Agent(genome),0])
        for i in range(1):
            genome = {
                'game_over': 189.27613725914273,
                'survival_instinct': 8.388926084018738,
                'total_height': -0.17634932529980674,
                'lines_removed': 8.594602383216944,
                'holes': -4.743561101942274,
                'bumpiness': -6.683915232551735,
                'pillar': -11.042880500059761,
                'y_pos_reward': 207.81525814829266,
                'y_pos_punish': 117.90325502640637
            }
            population.append([genome,Agent(genome),0])
        return population

    def save_population_to_file(self, population, filename='./data/population.txt'):
        A = sorted(population, key=lambda x: x[2])[min(100, POPULATION_SIZE//2):]
        with open(filename, "a") as file:
            for genome,_,score in A:
                file.write(f'{[genome,score]}\n')
        print(f'Population saved successfully to {filename}')

    def best_half(self, population):
        population = list(enumerate(population))
        population.sort(key=lambda x: (-x[1][2], x[0]))
        print(f'best in generation {self.generation_passed} = {population[0][1][2]}')
        # return [[genome[1][0],genome[1][2]] for genome in population[:POPULATION_SIZE//2]]
        return [[genome[1][0],genome[1][2]] for genome in population[:10]]

    def selection(self, population):
        N = SIZE_PICK + ((self.generation_passed-1)//GENERATION_RATE)

        parent1 = max(sample(population, N),key=lambda x:x[1])[0]
        parent2 = max(sample(population, N),key=lambda x:x[1])[0]
        return parent1, parent2

    def get_crossover_rates(self, k=0.1, midpoint=50):
        # Uniform crossover rate using S-curve transition
        uniform_rate = 1 / (1 + np.exp(k * ((self.generation_passed-1) - midpoint)))
        # Alpha crossover is the complement of uniform
        alpha_rate = 1 - uniform_rate
        return uniform_rate, alpha_rate

    def crossover(self, parent1, parent2):
        u, a = self.get_crossover_rates()

        if random.random() < u:
            child = {key: parent1[key] if random.random() < 0.5 else parent2[key] for key in parent1}
        else:
            alpha = uniform(0, 1)
            child = {key: alpha * parent1[key] + (1 - alpha) * parent2[key] for key in parent1}

        return child

    def get_mutate_rate(self, initial_rate=0.50, min_rate=0.2, decay_start=100, k=0.08):
        if self.generation_passed < decay_start:
            return initial_rate
        else:
            return min_rate + (initial_rate - min_rate) * math.exp(-k * (self.generation_passed+1 - decay_start))

    def mutate(self, individual):
        for key in individual.keys():
            if np.random.rand() < self.get_mutate_rate():
                individual[key] = np.random.uniform(WEIGHT[key][0],WEIGHT[key][1])
        return individual

    def check_hiscore(self, population):
        best = max(population,key=lambda x:x[2])
        avg_lines = best[2]
        agent = best[1]
        hiscore = agent.hiscore
        if avg_lines > hiscore*1.10:
            print(f"Model saved because a new avg_lines was achieved. old={agent.hiscore} new={avg_lines}")
            agent.save_model(agent.model1)
            agent.save_hiscore(avg_lines)

    def run(self):
        population = self.create_population()
        last_hiscore = last_avg_lines = 0
        for generation in range(GENERATIONS):
            print(f'Currently on generation={self.generation_passed}')

            data = [population, self.TOTAL_GAMES, self.generation_passed, self.get_crossover_rates(), self.get_mutate_rate(),last_hiscore,last_avg_lines]

            population = self.main_screen.run2(*data)

            self.check_hiscore(population)

            last_hiscore = max(self.main_screen.games.values(),key=lambda x:x.hiscore).hiscore
            last_avg_lines = max(self.main_screen.games.values(),key=lambda x:x.total_lines).total_lines/TOTAL_GAMES

            # self.save_population_to_file(population)

            population = self.best_half(population)

            new_population = [[genome,Agent(genome),0] for genome,_ in population]

            while len(new_population) != POPULATION_SIZE:
                parent1, parent2 = self.selection(population)

                child = self.crossover(parent1, parent2)

                child = self.mutate(child)

                new_population += [[child,Agent(child),0]]

            population = new_population
            self.generation_passed += 1
            print('')

        return population

def run_game():
    GA = Genetic_Algo()
    GA.run()

if __name__=='__main__':
    # GA = Genetic_Algo()
    # GA.run()
    run_game()
    # cProfile.run('run_game()', 'profile_output.prof')
    # p = pstats.Stats('profile_output.prof')
    # p.strip_dirs().sort_stats('cumulative').print_stats(lambda x: x >= 1)