import random
from prioritized_memory import PrioritizedMemory
from model import QTrainer, Linear_QNet
import numpy as np
import torch
import os


class Agent:
    def __init__(self, data):
        self.MAX_MEMORY = data[0]
        self.STATES = data[1]
        self.HIDDEN_AMOUNT = data[2]
        self.ACTIONS = data[3]
        self.BATCH_SIZE = data[4]
        self.LR = data[5]
        self.EPOCH = data[6]
        self.n_games = 0
        self.epsilon = 0.3  # randomness
        self.gamma = 0.999 # discount rate
        self.memory = PrioritizedMemory(self.MAX_MEMORY)

        self.model1 = Linear_QNet(self.STATES,self.HIDDEN_AMOUNT,self.ACTIONS) # primary network, evaluates best action
        self.model2 = Linear_QNet(self.STATES,self.HIDDEN_AMOUNT,self.ACTIONS) # target network,  evaluates best action's q value
        self.trainer = QTrainer(self.model1, self.model2, lr=self.LR, gamma=self.gamma, memory=self.memory,EPOCH=self.EPOCH,BATCH_SIZE=self.BATCH_SIZE)

        self.model_path = "model/best_model.pth"
        self.load_model(self.model_path)


        self.total_steps = 0
        self.losses = []
        self.q_values = []
        self.random = False
        self.total_games = data[7] - 9500
        #self.min_num = 1e-4
        self.min_num = 0.0001

        self.epsilon_0 = self.epsilon
        self.alpha = 1  # Controls how quickly the decay accelerates

    def clear_memory(self):
        self.memory = PrioritizedMemory(self.MAX_MEMORY)

    def load_model(self, model_path):
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)

            # Load model weights
            self.model1.load_state_dict(checkpoint['model_state_dict'])
            self.model2.load_state_dict(checkpoint['model_state_dict'])  # Target model sync

            # Load optimizer state
            self.trainer.optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'Model Loaded {model_path}')

        else:
            print("No saved model found. Using a new model.")

    def save_model(self):
        os.makedirs("model", exist_ok=True)  # Ensure directory exists
        checkpoint = {
            'model_state_dict': self.model1.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer1.state_dict()
        }
        torch.save(checkpoint, "model/best_model.pth")
        print("Model and optimizer saved successfully!")

    def calculate_lr(self, games_played: int, max_games: int = 500, min_lr: float = 0.001, max_lr: float = 0.01):
        # # Ensure the learning rate doesn't drop below the minimum
        # lr = max_lr - (max_lr - min_lr) * (games_played / max_games)
        # lr = max(lr, min_lr)
        # self.LR = lr
        # self.trainer.update_lr(lr)

        # # Compute learning rate decay
        current_phase = (games_played // 1000)  # Determines if we're in phase 1, 3, etc.
        phase_start = (current_phase * 1000)   # Start of the phase (0, 1000, etc.)

        lr = max_lr - (max_lr - min_lr) * ((games_played - phase_start) / max_games)
        lr = max(lr, min_lr)
        self.LR = lr
        self.trainer.update_lr(lr)

    def clear_info(self):
        self.losses = []
        self.q_values = []

    def decay_epsilon(self, game_number, max_games=500):
        # epsilon_t = self.min_num + (self.epsilon_0 - self.min_num) * (1 - (game_number / self.total_games)) ** self.alpha
        # self.epsilon = max(0.0001, epsilon_t)

        # Determine which phase we are in
        current_phase = (game_number // 1000)  # Phase switches at 1000, 2000, etc.
        phase_start = current_phase * 1000 # Start of current phase (0, 1000, 2000, etc.)

        if game_number == phase_start:
            self.epsilon = self.epsilon_0  # Reset to initial value

        # Apply decay only within the first `max_games` of each phase
        if (game_number - phase_start) <= max_games:
            epsilon_t = self.min_num + (self.epsilon_0 - self.min_num) * (1 - ((game_number-phase_start) / max_games)) ** self.alpha
            self.epsilon = max(0.0001, epsilon_t)

    def remember(self, state, next_state, reward, finished):
        # Get the Q-value for the current state from the primary model (for the action taken)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        current_q_value = self.model1(state_tensor)

        # For non-terminal states, calculate the target Q-value:
        if not finished:
            # Use the target network to estimate the Q-value for the next state (using max_a Q(s', a))
            next_q_value = torch.max(self.model2(next_state_tensor)).item()
            # The target is the immediate reward plus the discounted value of the next state
            target_q_value = reward + self.gamma * next_q_value
        else:
            # If the game ends (terminal state), just use the reward
            target_q_value = reward

        # Calculate the TD error (difference between target and current Q-value)
        td_error = abs(target_q_value - current_q_value)

        # Add the experience and TD error to memory
        self.memory.add((state, next_state, reward, finished), td_error)

    def train_long_memory(self):
        self.clear_info()
        loss = self.trainer.train()
        self.losses.append(loss)

    def update_target_network(self):
        self.model2.load_state_dict(self.model1.state_dict())

    def get_action(self, states):
        self.total_steps += 1
        if self.total_steps%1000==0:
            self.update_target_network()
        # Exploration vs exploitation
        if random.random() < self.epsilon:
            self.random = True
            return random.choice(list(states))
        else:
            self.random = False

        # Convert the dictionary keys (tuples) into a numpy array of floats
        state_values = np.array([list(state) for state in states], dtype=np.float32)

        # Convert the numpy array into a PyTorch tensor (only once)
        state_tensors = torch.from_numpy(state_values)

        with torch.no_grad():
            q_values = self.model1(state_tensors)

        self.q_values += [torch.max(q_values).item()]
        # Choose the state with the highest Q-value
        best_idx = torch.argmax(q_values).item()
        return list(states)[best_idx]

