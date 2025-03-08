from model import *

LR = 0.01
STATES = 6
HIDDEN_SIZES = [32,32,32]
ACTIONS = 1

MAX_MEMORY = 10000
BATCH_SIZE = 128
EPOCHS = 2
class Agent:
    def __init__(self, genome):
        self.weight = genome
        self.MAX_MEMORY = MAX_MEMORY
        self.STATES = STATES
        self.HIDDEN_AMOUNT = HIDDEN_SIZES
        self.ACTIONS = ACTIONS
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.EPOCH = EPOCHS
        self.n_games = 0
        self.epsilon = 0.3  # randomness
        self.gamma = 0.999 # discount rate
        self.memory = PrioritizedMemory(self.MAX_MEMORY)
        self.model1 = Linear_QNet(self.STATES,self.HIDDEN_AMOUNT,self.ACTIONS) # primary network, evaluates best action
        self.model2 = Linear_QNet(self.STATES,self.HIDDEN_AMOUNT,self.ACTIONS) # target network,  evaluates best action's q value
        self.trainer = QTrainer(self.model1, self.model2, lr=self.LR, gamma=self.gamma, memory=self.memory,EPOCH=self.EPOCH,BATCH_SIZE=self.BATCH_SIZE)

        self.model_path = "model/best_model.pth"
        self.load_model()

        self.total_steps = 0
        self.losses = []
        self.q_values = []
        self.random = False
        self.trained = False
        self.games_without_training = 0
        self.total_games = 500

        self.min_num = 0.0001
        self.decay_rate = (self.min_num / self.epsilon) ** (1 / self.total_games)

        self.epsilon_0 = self.epsilon
        self.alpha = 1  # Controls how quickly the decay accelerates

        self.hiscore_file = "./data/hiscore.txt"
        self.hiscore = self.load_hiscore()

    def load_hiscore(self):
        if os.path.isfile(self.hiscore_file):
            with open(self.hiscore_file, "r") as f:
                return float(f.read().strip())
        else:
            return 0.0

    def save_hiscore(self, score):
        with open(self.hiscore_file, "w") as f:
            f.write(str(score))

    def save_model(self, model):
        os.makedirs("./model", exist_ok=True)  # Ensure directory exists
        checkpoint = {
            'model_state_dict': self.model1.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer1.state_dict()
        }
        torch.save(checkpoint, "model/best_model.pth")
        print("Model and optimizer saved successfully!")

    def load_model(self):
        if os.path.isfile(self.model_path):
            checkpoint = torch.load(self.model_path)

            # Load model weights
            self.model1.load_state_dict(checkpoint['model_state_dict'])
            self.model2.load_state_dict(checkpoint['model_state_dict'])  # Target model sync

            # Load optimizer state
            self.trainer.optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])

        else:
            print("No saved model found. Using a new model.")

    def calculate_lr(self, games_played: int, max_games: int = 500, min_lr: float = 0.001, max_lr: float = 0.01):
        lr = max_lr - (max_lr - min_lr) * (games_played / max_games)
        lr = max(lr, min_lr)
        self.LR = lr
        self.trainer.update_lr(lr)

    def decay_epsilon(self, game_number):
        # self.epsilon = max(0.04, self.epsilon * 0.977) # 100
        # Apply exponential decay
        epsilon_t = self.min_num + (self.epsilon_0 - self.min_num) * (1 - (game_number / self.total_games)) ** self.alpha
        self.epsilon = max(self.min_num, epsilon_t)
        return

    def remember(self, state, next_state, reward, finished):
        # # Get the Q-value for the current state from the primary model (for the action taken)
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32)

        current_q_value = self.model1(state_tensor).squeeze().item()

        with torch.no_grad():
            # For non-terminal states, calculate the target Q-value:
            if not finished:
                # Use the target network to estimate the Q-value for the next state (using max_a Q(s', a))
                next_q_value = self.model2(next_state_tensor).max().item()
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
        # self.clear_info()
        self.trainer.clear_q()
        loss = self.trainer.train()
        #self.decay_epsilon()
        #self.losses.append(loss)

    def update_target_network(self):
        self.model2.load_state_dict(self.model1.state_dict())

    def get_action(self, states):
        if random.random() < self.epsilon:
            self.random = True
            return random.choice(list(states))
        else:
            self.random = False

        # Convert the dictionary keys (tuples) into a numpy array of floats
        state_values = np.array([list(state) for state in states], dtype=np.float32)
        # Convert the numpy array into a PyTorch tensor (only once)
        #state_tensors = torch.from_numpy(state_values)
        state_tensors = torch.as_tensor(state_values)

        with torch.no_grad():
            q_values = self.model1(state_tensors)

        #self.q_values += [torch.max(q_values).item()]
        # Choose the state with the highest Q-value
        best_idx = torch.argmax(q_values).item()
        # return list(states)[best_idx]
        states_list = list(states)
        return states_list[best_idx]

    def check_steps(self):
        self.total_steps += 1
        if self.total_steps%1000==0:
            self.update_target_network()
        if self.total_steps%200==0:
            self.train_long_memory()
            self.trained = True

    def check_training(self):
        # train at least once a game
        if not self.trained:
            self.train_long_memory()
            self.games_without_training += 1
            # 5 games without training
            if self.games_without_training==5:
                self.update_target_network()
                self.games_without_training = 0
        else:
            self.games_without_training = 0
        self.trained = False

    def calculate_rewards(self,best_state, finished):
        total_heights, bumpiness, lines_removed, holes, y_pos, pillar = best_state
        calc_reward = 0

        # Define when the board is "half-full"
        board_half_full = total_heights >= 110 or (total_heights >= 90 and bumpiness >= 10)

        # hole_penalty = self.weight['holes']
        if total_heights >= 140 or (total_heights >= 110 and bumpiness >= 12):
            hole_penalty = -2.743561101942274  # Reduced penalty when board is high
        elif total_heights >= 90 or (total_heights >= 70 and bumpiness >= 9):
            hole_penalty = -4.743561101942274
        else:
            hole_penalty = self.weight['holes']

        pillar_penalty = 0
        if holes > 0 or board_half_full:
            pillar_penalty = self.weight['pillar']

        # Discourage Placing High When Board is Low
        if total_heights <= 40:  # Board is mostly empty
            high_placement_penalty = (10 - y_pos) * 2  # Stronger penalty
        elif total_heights <= 100:  # Board is partially filled
            high_placement_penalty = (10 - y_pos)  # Moderate penalty
        else:
            high_placement_penalty = 0  # No penalty when the board is high

        if y_pos >= 12:  # If the piece is placed in the upper 40% of the board
            calc_reward -= high_placement_penalty

        # Game over penalty
        if finished:
            calc_reward -= self.weight['game_over']  # Severe punishment for game over

        # Base survival incentive
        calc_reward += self.weight['survival_instinct']

        # Low piece placement reward (encourage low stacking)
        if y_pos >= 9:
            calc_reward += self.weight['y_pos_reward']  # Reward for stacking high when appropriate
        else:
            calc_reward -= (10 - y_pos) * 0.2 - self.weight['y_pos_punish']  # Gradual penalty for high stacking

        # Penalty for total board height
        calc_reward += self.weight['total_height'] * total_heights  # Encourage keeping the board low

        # Line clear reward
        calc_reward += (2 ** lines_removed) * self.weight['lines_removed']  # Scaled reward for big clears
        if lines_removed == 4:
            calc_reward += 5000

        # Penalty for holes
        calc_reward += hole_penalty * holes

        # Penalty for bumpiness (prefer smooth board)
        calc_reward += self.weight['bumpiness'] * bumpiness

        # Pillar penalty
        calc_reward += pillar_penalty

        return calc_reward

class PrioritizedMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []  # Store as list of (priority, counter, experience)
        self.counter = 0

    def add(self, experience, priority):
        # Detach tensor priority and convert it to np.float32
        if isinstance(priority, torch.Tensor):
            priority = priority.detach().cpu().numpy().item()  # Convert tensor to a scalar value
        else:
            priority = np.float32(priority)

        if len(self.memory) < self.max_size:
            heapq.heappush(self.memory, (priority, self.counter, experience))
        else:
            heapq.heappushpop(self.memory, (priority, self.counter, experience))
        self.counter += 1  # Increment the counter for the next experience

    def sample(self, batch_size, beta=0.4):
        # Normalize priorities to sum up to 1
        priorities = np.array([priority for priority, _, _ in self.memory], dtype=np.float32)

        probabilities = priorities ** beta
        probabilities /= probabilities.sum()

        # Sample based on probability distribution using the fast_sample function
        indices = fast_sample(batch_size, probabilities)
        indices = np.clip(indices, 0, len(probabilities) - 1)

        # Collect the batch using the indices
        batch = [self.memory[i][2] for i in indices]

        # Calculate importance-sampling weights
        weights = (1.0 / len(self.memory) / probabilities[indices]) ** beta
        weights /= weights.max()

        return batch, indices, weights

    def update_priority(self, indices, td_errors):
        for i, idx in enumerate(indices):
            self.memory[idx] = (np.float32(td_errors[i].item()), self.memory[idx][1], self.memory[idx][2])
        heapq.heapify(self.memory)

    def fast_sample(self, batch_size: int, probabilities: np.ndarray) -> np.ndarray:
        # Generate cumulative sum of probabilities (like a CDF)
        cumulative_probs = np.cumsum(probabilities)
        random_vals = np.random.rand(batch_size)

        # For each random value, find the corresponding index using binary search
        indices = np.searchsorted(cumulative_probs, random_vals)
        return indices

    def __len__(self):
        return len(self.memory)
