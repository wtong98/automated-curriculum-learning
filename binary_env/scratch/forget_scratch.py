"""
Studying the forget dynamics
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../')

from env import *

# <codecell>
def run_sim(N, eps, max_iters=1000, big_reward=1e4):
    student = Student(lr=1, q_e=eps)
    env = BinaryEnv(N, reward=big_reward)
    student.learn(env)
    env = BinaryEnv(N + 1, reward=big_reward)

    total = 0
    for _ in range(max_iters):
        is_done = False

        state = env.reset()
        while not is_done:
            a = student.next_action(state)
            new_state, reward, is_done, _ = env.step(a)
            student.update(state, a, reward, new_state, is_done)
            state = new_state

        total += 1
        if 0 not in student.q_r.values():
            break

    return total
# %%
n_iters = 10000
avg = 0
for _ in tqdm(range(n_iters)):
    avg += run_sim(5, 1) / n_iters

print(avg)

# %%
def sig(x):
    return 1 / (1 + np.exp(-x))

# %%
