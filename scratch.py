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

N = 6
agent = Student(lr=0.1)
scores = []
qs = []
max_iters = 100

def log(agent):
    scores.append(agent.score(N))
    qs.append(agent.q_r)

agent.learn(BinaryEnv(N, reward=10), max_iters=max_iters, post_hook=log)
print('done')

# %%
plt.plot(scores)
# %%
plt.bar([0,1,2,3,4,5], height=list(qs[-1].values()))

# <codecell> IMPROMPTU TEACHER TESTING
N = 10
T = 100
bins = 20
teacher = Teacher(bins=bins)

i = 0

path = []
completions = []

def log(teacher):
    global i
    i += 1
    path.append((env.N, env.student.score(env.N)))

def done(teacher):
    global i
    completions.append(i-1)

env = CurriculumEnv(N, T, p_eps=0.1, teacher_reward=10, student_reward=10)
teacher.learn(env, max_iters=10000, use_tqdm=True, post_hook=log, done_hook=done)

path = np.array(path)
print('done!')

# TODO: debug teacher / get q-values > 0
# TODO: reset student after successful episode
# %%
fig, axs = plt.subplots(2, 1, figsize=(8, 5))

axs[0].plot(path[:,0])
for c in completions:
    if c == completions[0]:
        axs[0].axvline(x=c, color='red', alpha=0.3, label='Episode boundary')
    else:
        axs[0].axvline(x=c, color='red', alpha=0.3)

axs[0].set_title('Teacher')
axs[0].set_xlabel('time')
axs[0].set_ylabel('N')
axs[0].legend()

axs[1].plot(path[:,1])
for c in completions:
    if c == completions[0]:
        axs[1].axvline(x=c, color='red', alpha=0.3, label='Episode boundary')
    else:
        axs[1].axvline(x=c, color='red', alpha=0.3)

axs[1].set_title('Student')
axs[1].set_xlabel('time')
axs[1].set_ylabel('Log prob')
axs[1].legend()

fig.tight_layout()
plt.savefig('fig/teacher_train.png')

# %%
plt.plot(path[:,1])

# %%
def _make_heatmap(action_idx):
    im = np.zeros((N, 21))

    for i, row in enumerate(np.arange(N) + 1):
        for j, col in enumerate(np.arange(0, 1.01, step=0.05)):
            im[i, j] = teacher.q[(row, col), action_idx]
    
    return im

ims = [_make_heatmap(i) for i in [0, 1, 2]]

fig, axs = plt.subplots(3, 1, figsize=(5, 6))
for i, (im, ax) in enumerate(zip(ims, axs.ravel())):
    m = ax.imshow(im, vmin=0, vmax=5)

    ax.set_title(f'Action: {i-1}')
    ax.set_xlabel('Probability of success')
    ax.set_ylabel('N')
    fig.colorbar(m, ax=ax)

fig.suptitle('Teacher Q values')
fig.tight_layout()
plt.savefig('fig/teacher_q.png')


# %%
### PLOT PHASE DIAGRAM OF STRATEGY
ll, nn = np.meshgrid(np.linspace(0, 1, num=bins), np.arange(1, N + 1))
actions = []

for l, n in zip(ll.ravel(), nn.ravel()):
    a = teacher.next_action((n, l))
    actions.append(a)

z = np.array(actions).reshape(ll.shape) - 1
plt.contourf(ll, nn, z)
plt.colorbar()

# %%
