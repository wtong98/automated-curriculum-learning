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
def run_sim(N, eps, max_iters=1000, lr=1, reward=1e4):
    student = Student(lr=lr, q_e=eps)
    env = BinaryEnv(N, reward=reward)
    student.learn(env)
    env = BinaryEnv(N + 1, reward=reward)

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
        # if 0 not in student.q_r.values():
        #     break
        if student.score(N+1) > -0.05:
            break

    return total

def pred_iters(N, eps):
    return 1 / sig(eps) ** (N+1) + N

def sig(x):
    return 1 / (1 + np.exp(-x))
# %%
n_iters = 1000
eps = 0
N = 5

samps = []
for _ in tqdm(range(n_iters)):
    samps.append(run_sim(N, eps, lr=0.1, reward=10))
    # samps.append(run_sim(N, eps, lr=1, reward=1000))

# %%
plt.hist(samps, bins=30)
plt.axvline(np.mean(samps), color='black', label='samp mean')
plt.axvline(pred_iters(N, eps), color='red', linestyle='dashed', alpha=0.8, label='pred mean')
plt.legend()

# %%
