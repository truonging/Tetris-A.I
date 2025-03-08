import matplotlib.pyplot as plt
from IPython import display
import numpy as np

plt.ion()
fig, mat = plt.subplots(1,4,figsize=(12,5))
def plot(avg_losses, rewards, q_choose_confidence, q_policy_strength, lines):
    A,B,C,D = mat[0],mat[1],mat[2],mat[3]
    A.cla()
    B.cla()
    C.cla()
    D.cla()


    A.plot(avg_losses, marker='o', linestyle='-', color='r')
    A.set_title("Average Losses")

    moving_avg = np.convolve(rewards, np.ones(100) / 100, mode='same')
    B.plot(rewards, marker='o', linestyle='-', color='g')
    B.plot(moving_avg, label="Moving Average", color='orange')
    B.set_title("Average Rewards")

    C.plot(q_choose_confidence, marker='o', linestyle='-', color='b')
    C.plot(q_policy_strength, marker='o', linestyle='-', color='c')
    C.set_title("Q_Values")

    moving_avg2 = np.convolve(lines, np.ones(100) / 100, mode='same')
    D.plot(lines, marker='o', linestyle='-', color='m')
    D.set_title('Total Lines')
    D.plot(moving_avg2, label="Moving Average", color='orange')


    mn = min(min(q_choose_confidence), min(q_policy_strength))
    mx = max(max(q_choose_confidence), max(q_policy_strength))
    A.set_ylim(min(avg_losses),max(avg_losses))
    B.set_ylim(min(rewards),max(rewards))
    C.set_ylim(mn, mx)

    # Show grid
    A.grid(True)
    B.grid(True)
    C.grid(True)
    D.grid(True)

    # Update the figure and pause
    display.clear_output(wait=True)
    plt.draw()  # Redraw the figure with updated data
    plt.pause(0.1)

    # # display.clear_output(wait=True)
    # # display.display(plt.gcf())
    # # plt.clf()
    # # plt.title("Scores Over Games")
    # # plt.xlabel("Games")
    # # plt.ylabel("Score")
    # # plt.plot(scores, marker='o', linestyle='-', color='b')
    # # plt.ylim(ymin=0)
    # # plt.xticks(range(len(scores)))  # Ensure x-axis ticks are whole numbers
    # # plt.grid(True)
    # # plt.show(block=False)
    # # plt.pause(0.1)
    # # Plot loss
    # display.clear_output(wait=True)
    # #display.display(plt.gcf())
    # fig.clf()
    # print(avg_losses)
    # print(rewards)
    # A.plot(avg_losses, marker='o', linestyle='-', color='r')
    # B.plot(rewards, marker='o', linestyle='-', color='g')
    # #A.plot(q_choose_confidence, marker='o', linestyle='-', color='b')
    # #B.plot(q_policy_strength, marker='o', linestyle='-', color='c')
    # # plt.xlabel("Episode")
    # # plt.title("Data Over Episodes")
    # # plt.xticks(range(len(rewards)))
    # plt.ylim(min(min(avg_losses),min(rewards)), max(max(avg_losses),max(rewards)))
    # plt.grid(True)
    # fig.show()
    # plt.pause(0.1)

"""

Metric	               What It Shows	  Desired Trend
Training Loss	     Model convergence	     Downward
Episode Reward	    Gameplay performance	  Upward
Q-Value training	  Policy stability	   Stabilizing
Q-Value choose      decision confidence    Stabilizing
Lines                    Survival             Upward

"""