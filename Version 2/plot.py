import matplotlib.pyplot as plt
#from settings import *
import os
import numpy as np
import ast

plt.ion()  # Turn on interactive mode (this keeps the plot open)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))


def read_file(filepath):
    with open(filepath, "r") as file:
        return [line.strip() for line in file.readlines()]

def plot():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, 'data')

    # File paths
    file_paths = {
        "all_generations": os.path.join(data_folder, "all_generations.txt"),
        "generation_avg_lines": os.path.join(data_folder, "generation_mean_line.txt"),
        "generation_std_dev": os.path.join(data_folder, "generation_std_dev.txt"),
        "avg_diversity": os.path.join(data_folder, "diversity.txt")
    }

    # Read each file and store contents in a list
    all_generations = read_file(file_paths["all_generations"])
    generation_avg_lines = read_file(file_paths["generation_avg_lines"])
    generation_std_dev = read_file(file_paths["generation_std_dev"])
    avg_diversity = read_file(file_paths["avg_diversity"])

    # Convert numerical data to floats (except all_generations, which might be strings)
    all_generations = [ast.literal_eval(gen) for gen in all_generations][65:]
    generation_avg_lines = [float(x) for x in generation_avg_lines][65:]
    generation_std_dev = [float(x) for x in generation_std_dev][65:]
    avg_diversity = [float(x) for x in avg_diversity][65]

    # Clear the current axes to update the plot without overlapping
    # self.ax.clear()

    generations = len(all_generations)

    # --- Plot 1: Avg Lines per Game for Each AI and Generation Average ---
    # Plot the scatter plot for the current generation's AI performance
    for gen in range(1,generations+1):
        avg_lines_per_ai = all_generations[gen-1]  # Scores of all AIs in generation `gen`
        ax1.scatter([gen] * 200, avg_lines_per_ai, alpha=0.7, color='g', s=10)
    #ax1.scatter([gen] * POPULATION_SIZE, all_generations[-1], alpha=0.7, color='g', s=10)

    # Plot the average lines for each generation (red line)
    ax1.plot(range(1, generations + 1), generation_avg_lines, marker='o', linestyle='-', color='r')

    # Formatting the scatter plot
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Avg Lines per Game')
    ax1.set_title('AI Performance: Avg Lines/Game for Each Agent and Generation Average')
    ax1.set_ylim(0, max(generation_avg_lines) + 10)  # Keep y-axis consistent and dynamic
    ax1.set_xticks(np.arange(1, generations + 1, 1))  # Set x-axis ticks to be whole numbers from 1 to current gen

    # --- Plot 2: Diversity Over Generations ---
    # Plot diversity (blue line)
    ax2.plot(range(1, generations + 1), avg_diversity, marker='o', linestyle='-', color='b',
                  label='Diversity')

    # Formatting the diversity plot
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Diversity')
    ax2.set_title('Genetic Diversity Over Generations')
    ax2.set_xticks(np.arange(1, generations + 1, 1))  # Set x-axis ticks to be whole numbers from 1 to current gen

    # Pause to update the plot
    plt.pause(0.1)  # You can adjust the pause time if needed
    plt.show(block=True)  # Keep the plot window open after the loop finishes

if __name__=='__main__':
    plot()
