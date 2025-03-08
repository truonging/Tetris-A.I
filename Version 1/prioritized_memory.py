import heapq
import torch
from njit_startup import *


class PrioritizedMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []  # Store as list of (priority, counter, experience)
        self.counter = 0

    def add(self, experience, priority):
        if isinstance(priority, torch.Tensor):
            priority = priority.detach().cpu().numpy().item()
        else:
            priority = np.float32(priority)

        if len(self.memory) < self.max_size:
            heapq.heappush(self.memory, (priority, self.counter, experience))
        else:
            heapq.heappushpop(self.memory, (priority, self.counter, experience))
        self.counter += 1

    def sample(self, batch_size, beta=0.4):
        # Normalize priorities to sum up to 1
        priorities = np.array([priority for priority, _, _ in self.memory], dtype=np.float32)

        probabilities = priorities ** beta
        probabilities /= probabilities.sum()

        # Sample based on probability distribution
        indices = fast_sample(batch_size, probabilities)
        indices = np.clip(indices, 0, len(probabilities) - 1)

        batch = [self.memory[i][2] for i in indices]

        # Calculate importance sampling weights
        weights = (1.0 / len(self.memory) / probabilities[indices]) ** beta
        weights /= weights.max()

        return batch, indices, weights

    def update_priority(self, indices, td_errors):
        for i, idx in enumerate(indices):
            self.memory[idx] = (np.float32(td_errors[i].item()), self.memory[idx][1], self.memory[idx][2])
        heapq.heapify(self.memory)

    def __len__(self):
        return len(self.memory)
