"""
Teacher and student agents

author: William Tong (wtong@g.harvard.edu)
"""

from collections import defaultdict

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# TODO: figure out specifics to abstract
class Agent:
    def __init__(self) -> None:
        pass
    
    # def next_action(state) -> object:
    #     pass


class Student(Agent):
    def __init__(self, lr, gamma=1, q_e=None) -> None:
        super().__init__()
        self.lr = lr
        self.gamma = gamma

        # only track Q-values for action = 1, maps state --> value
        self.q_e = defaultdict(int) if q_e == None else q_e
        self.q_r = defaultdict(int)
    
    # softmax policy
    def policy(self, state) -> np.ndarray:
        q = self.q_e[state] + self.q_r[state]
        prob = sigmoid(q)
        return np.array([1 - prob, prob])

    def next_action(self, state) -> int:
        _, prob = self.policy(state)
        return np.random.binomial(n=1, p=prob)
    
    def update(self, old_state, reward, next_state):
        _, prob = self.policy(next_state)
        exp_q = prob * self.q_r[next_state]
        self.q_r[old_state] += self.lr * (reward + self.gamma * exp_q - self.q_r[old_state])
