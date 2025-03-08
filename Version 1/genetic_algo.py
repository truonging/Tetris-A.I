import numpy as np

from train import Training_Simulation
from random import sample, uniform, random

import torch.multiprocessing as mp
import ast
from collections import deque

FILE_NAME = 'population.txt'
POPULATION_SIZE = 5
SIZE_PICK = 3
TOTAL_GAMES = 10000
GAMES_RATE = 100
GENERATIONS = 1000
GENERATION_RATE = 100

WEIGHT = {
    'game_over': (-1000,-100),
    'survival_instinct': (1,50),
    "total_height": (-20, 0),
    "lines_removed": (1, 20),
    "holes": (-20, 0),
    "bumpiness": (-20, 0),
    "pillar": (-100, 0),
    'y_pos_reward': (50,300),
    'y_pos_punish': (50,300)
}
"""
1000 generation
random population of 100
50/50 move on       priority: lines -> survival
tournment selection for offsprings
uniform 100% -> 0%     (10% decrement every 100) generation
alpha   0% -> 100%     (10% increment every 100) generation
mutate  20% -> 5%      (1.5% decrement every 100) generation
games   50 -> 550      (50 increment every 100) generation
pick    3 -> 8         (1 increment every 100*2) generation

epsilon 0.3 -> 0.01       (0.3 -> 0.01 in exactly games) games
decay rate based on total games
LR      0.01 -> 0.001  (decrement every game til 500) games
gamma   0.97
"""
class Genetic_Algo:
    def __init__(self):
        self.generation_passed = 1
        self.tetris_hiscore = 0

    def create_population(self,n=POPULATION_SIZE):
        return [
            [{
                'game_over': np.random.uniform(WEIGHT['game_over'][0], WEIGHT['game_over'][1]),
                'survival_instinct': np.random.uniform(WEIGHT['survival_instinct'][0], WEIGHT['survival_instinct'][1]),
                "total_height": np.random.uniform(WEIGHT['total_height'][0], WEIGHT['total_height'][1]),
                "lines_removed": np.random.uniform(WEIGHT['lines_removed'][0], WEIGHT['lines_removed'][1]),
                "holes": np.random.uniform(WEIGHT['holes'][0], WEIGHT['holes'][1]),
                "bumpiness": np.random.uniform(WEIGHT['bumpiness'][0], WEIGHT['bumpiness'][1]),
                "pillar": np.random.uniform(WEIGHT['pillar'][0], WEIGHT['pillar'][1]),
                'y_pos_reward': np.random.uniform(WEIGHT['y_pos_reward'][0], WEIGHT['y_pos_reward'][1]),
                'y_pos_punish': np.random.uniform(WEIGHT['y_pos_punish'][0], WEIGHT['y_pos_punish'][1])
            }, 0]
            # [{'game_over': 189.27613725914273,
            #   'survival_instinct': 8.388926084018738,
            #   'total_height': -0.17634932529980674,
            #   'lines_removed': 8.594602383216944,
            #   'holes': -10.743561101942274,
            #   'bumpiness': -6.683915232551735,
            #   'pillar': -11.042880500059761,
            #   'y_pos_reward': 207.81525814829266,
            #   'y_pos_punish': 117.90325502640637
            # },0]
            for _ in range(n)
        ]

    def fetch_population(self):
        filename = FILE_NAME

        with open(filename, 'r') as file:
            last_lines = deque(file, maxlen=POPULATION_SIZE)
        res = []
        for line in last_lines:
            data = ast.literal_eval(line.strip())
            res += [[data[0],0]]
        return res

    def fitness(self, a):
        (individual,i) = a
        total_games = TOTAL_GAMES #+ ((self.generation_passed // GAMES_RATE) * 50)
        sim = Training_Simulation(individual, i, self.generation_passed, total_games)
        hiscore, lines, tetris_clears = sim.run_simulation(total_games)
        #return lines/total_games, survival_time
        return float("{:.3f}".format(lines/total_games)), hiscore, sim.agent, tetris_clears

    def pick_elite(self,very_best=False):
        filename = FILE_NAME
        max_fitness = float('-inf')
        best_genome = None

        with open(filename, 'r') as file:
            lines = file.readlines()
        if not very_best:
            lines = lines[-POPULATION_SIZE:]
        for line in lines:
            data = ast.literal_eval(line.strip())
            genome, fitness = data[0], data[1]
            if fitness > max_fitness:
                max_fitness = fitness
                best_genome = genome

        print(f'elite of generation {self.generation_passed} = {max_fitness} {best_genome}')
        return [best_genome,0,0]

    def selection(self, population):
        N = SIZE_PICK + (self.generation_passed//(GENERATION_RATE*2))
        parent1 = max(sample(population, N),key=lambda x:x[1])[0]
        parent2 = max(sample(population, N),key=lambda x:x[1])[0]
        return parent1, parent2

    def get_crossover_rates(self, k=0.02, midpoint=600):
        # Uniform crossover rate using S-curve transition
        uniform_rate = 1 / (1 + np.exp(k * (self.generation_passed - midpoint)))
        # Alpha crossover is the complement of uniform
        alpha_rate = 1 - uniform_rate
        return uniform_rate, alpha_rate

    def crossover(self, parent1, parent2):
        # Get crossover rates
        u, a = self.get_crossover_rates()
        # Uniform crossover if roll is within uniform rate
        if random() < u:
            child = {key: parent1[key] if random() < 0.5 else parent2[key] for key in parent1}
        else:
            alpha = uniform(0, 1)
            child = {key: alpha * parent1[key] + (1 - alpha) * parent2[key] for key in parent1}

        return child

    def mutate(self, individual, mutation_rate=0.20):
        mutation_rate -= (self.generation_passed//100)*0.015
        for key in individual.keys():
            if np.random.rand() < mutation_rate:
                individual[key] = np.random.uniform(WEIGHT[key][0],WEIGHT[key][1])
        return individual

    def fitness_evaluation(self, population):
        BATCH_SIZE = 10
        fitness_results = []

        for batch_start in range(0, POPULATION_SIZE, BATCH_SIZE):
            batch = population[batch_start:batch_start + BATCH_SIZE]

            with mp.Pool(processes=min(mp.cpu_count(), BATCH_SIZE)) as pool:
                batch_results = pool.map(self.fitness, [(genome[0], i + batch_start) for i, genome in enumerate(batch)])

            fitness_results.extend(batch_results)

        # Update population with fitness scores
        for i in range(len(population)):
            population[i][1] = fitness_results[i][0]

        best = max(fitness_results, key=lambda x:(x[3]*300000)+x[1])
        if best[1] > self.tetris_hiscore:
            print(f'saving old={self.tetris_hiscore} new={best[1]} with {best[3]}')
            self.tetris_hiscore = best[1] + best[3]*300000
            best[2].save_model()

        return population

    def save_population_to_file(self, population):
        filename = FILE_NAME
        with open(filename, "a") as file:
            for genome in population:
                file.write(f"{genome}\n")
        print(f"Population saved successfully to {filename}")

    def best_half(self, population):
        population = list(enumerate(population))
        population.sort(key=lambda x: (-x[1][1], x[0]))
        for p in population:
            print(f'{p[1][1]}')
        print(f'best in generation {self.generation_passed} = {population[0][1][1]}')
        return [[genome[1][0],0] for genome in population[:POPULATION_SIZE//2]]

    def run(self):
        population = self.create_population()
        #population = self.fetch_population()

        for generation in range(GENERATIONS):
            print(f'currently on generation {self.generation_passed}')

            # evaluate fitness of entire population
            population = self.fitness_evaluation(population)

            #self.save_population_to_file(population)

            # selection (elitism + tournament)
            population = self.best_half(population)

            new_population = population[:]

            while len(new_population) != POPULATION_SIZE:
                parent1, parent2 = self.selection(population)

                # crossover (uniform + arithmetic)
                child = self.crossover(parent1, parent2)

                # mutate (20% per key)
                child = self.mutate(child)

                new_population += [[child,0]]

            population = new_population
            self.generation_passed += 1

        return population


if __name__=='__main__':
    algo = Genetic_Algo()
    for thing in algo.run():
        print(thing)