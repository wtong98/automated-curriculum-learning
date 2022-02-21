"""
Field for training agents and plotting results

author: William Tong (wtong@g.harvar.edu)
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np

from agent import Student, Teacher
from env import BinaryEnv, CurriculumEnv

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

# <codecell> IMPROMPTU TEACHER TESTING
teacher = Teacher()
student = Student()
N = 5
T = 1000

path = []

def log(teacher):
    path.append((env.N, student.score(env.N)))

env = CurriculumEnv(student, N, T, p_eps=0.1, teacher_reward=10, student_reward=10)
teacher.learn(env, max_iters=500, post_hook=log)

path = np.array(path)
print('done!')

# TODO: debug teacher / get q-values > 0
# TODO: reset student after successful episode
# %%
plt.plot(path[:,0])

# %%
plt.plot(path[:,1])

# %%
