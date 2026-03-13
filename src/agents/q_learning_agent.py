import numpy as np
from collections import defaultdict


def state_to_key(state):
    key = 0
    for v in state:
        key = key * 3 + int(v)
    return key


class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.99, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(9, dtype=np.float32))

    def select_action(self, state, available_actions):
        key = state_to_key(state)
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(available_actions))
        qs = self.Q[key]
        # mask unavailable actions
        masked = np.full_like(qs, -np.inf)
        for a in available_actions:
            masked[a] = qs[a]
        return int(int(np.argmax(masked)))

    def update(self, state, action,
               reward, next_state, done,
               next_available_actions=None):
        s = state_to_key(state)
        ns = state_to_key(next_state)
        q = self.Q[s][action]
        if done:
            target = reward
        else:
            # use next_available actions to mask invalid moves
            if next_available_actions is None:
                best_next = np.max(self.Q[ns])
            else:
                best_next = max(self.Q[ns][a] for a in next_available_actions)
            target = reward + self.gamma * best_next
        self.Q[s][action] += self.alpha * (target - q)
