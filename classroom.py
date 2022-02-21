"""
Field for training agents and plotting results

author: William Tong (wtong@g.harvar.edu)
"""

import matplotlib.pyplot as plt

from agent import Student
from env import BinaryEnv

# <codecell>
# Scratch routine to train student

N = 20
agent = Student(lr=0.1)
scores = []
qs = []
max_iters = 10000

def log(agent):
    scores.append(agent.score(N))
    qs.append(agent.q_r)

agent.learn(BinaryEnv(N, reward=50), max_iters=max_iters, post_hook=log)
print('done')

# %%
plt.plot(scores)
# %%
plt.bar([1,2,3,4,5], height=list(qs[-1].values()))