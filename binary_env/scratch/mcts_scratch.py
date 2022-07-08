"""
Explore performance of MCTS
"""

# <codecell>
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../')
from env import *

# <codecell>
def run_mcts(n_iters=500, eps=0, goal_length=3, T=50, lr=0.01, max_steps=500):
    teacher = TeacherPerfectKnowledge(goal_length=goal_length, T=T, student_qe=eps, student_lr=lr, n_iters=n_iters)
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=T, anarchy_mode=True, student_params={'lr': lr})
    traj = [env.N]
    env.reset()

    N = goal_length
    qr = np.zeros(N)
    prev_a = None

    for _ in range(max_steps):
        a = teacher.next_action(prev_a, qr)
        state, _, is_done, _ = env.step(a)
        traj.append(a)

        if is_done:
            break

        # print(f'action: {a}  state: {state}')
        prev_a = a
        qr = np.array([env.student.q_r[i] for i in range(N)])

    return traj


def run_incremental(eps=0, goal_length=3, T=50, lr=0.01, max_steps=500):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=T, student_params={'lr': lr})
    env.reset()
    traj = [env.N]

    score = env._get_score(1, train=False)
    for _ in range(max_steps):
        if -score < env.p_eps:
            action = 2
        else:
            action = 1
        
        (_, score), _, is_done, _ = env.step(action)
        traj.append(env.N)

        if is_done:
            break
    
    return traj

# <codecell>
n_iters = 5
N = 4
lr = 0.1
max_steps = 500
eps = -1.5

mc_iters = 1000

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

cases = [
    Case('Incremental', run_incremental, {'eps': eps, 'goal_length': N, 'lr': lr}, []),
    Case('MCTS', run_mcts, {'eps': eps, 'goal_length': N, 'lr': lr, 'n_iters': mc_iters}, []),
]

for _ in tqdm(range(n_iters)):
    for case in cases:
        case.runs.append(case.run_func(**case.run_params, max_steps=max_steps))

# <codecell>
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

for i, case in enumerate(cases):
    label = {'label': case.name}
    for run in case.runs:
        axs[0].plot(run, color=f'C{i}', alpha=0.7, **label)
        label = {}

axs[0].legend()
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('N')
axs[0].set_yticks(np.arange(N) + 1)

all_lens = [[len(run) for run in case.runs] for case in cases]
all_means = [np.mean(lens) for lens in all_lens]
all_serr = [2 * np.std(lens) / np.sqrt(n_iters) for lens in all_lens]
all_names = [case.name for case in cases]

axs[1].bar(np.arange(len(cases)), all_means, tick_label=all_names, yerr=all_serr)
axs[1].set_ylabel('Iterations')

fig.suptitle(f'Epsilon = {eps}')
fig.tight_layout()
# plt.savefig(f'fig/mcts_n_{N}_eps_{eps}_zoom.png')



# %% LONG COMPARISON PLOT
n_iters = 7
N = 5
lr = 0.1
max_steps = 1000
eps = np.arange(-2, 2.1, step=0.5)

mc_iters = 1000

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

all_cases = [
    (
        Case('Incremental', run_incremental, {'eps': e, 'goal_length': N, 'lr': lr}, []),
        Case('MCTS', run_mcts, {'eps': e, 'goal_length': N, 'lr': lr, 'n_iters': mc_iters}, []),
    ) for e in eps
]

for _ in tqdm(range(n_iters)):
    for cases in all_cases:
        for case in cases:
            case.runs.append(case.run_func(**case.run_params, max_steps=max_steps))

# <codecell>
cases = zip(*all_cases)
all_means = []
all_ses = []

for case_set in cases:
    curr_means = []
    curr_ses = []

    for case in case_set:
        run_lens = [len(run) for run in case.runs]
        curr_means.append(np.mean(run_lens))
        curr_ses.append(np.std(run_lens) / np.sqrt(n_iters))

    all_means.append(curr_means)
    all_ses.append(curr_ses)


width = 0.2
offset = np.array([-1, 0])
x = np.arange(len(eps))
names = ['Incremental', 'MCTS']

for name, off, mean, se in zip(names, width * offset, all_means, all_ses):
    plt.bar(x+off, mean, yerr=se, width=width, label=name)
    plt.xticks(x, labels=eps)
    plt.xlabel('Epsilon')
    plt.ylabel('terations')

plt.legend()
plt.title(f'Teacher performance for N={N}')
plt.tight_layout()
plt.savefig('../fig/mcts_performance_comparison.png')

# %%
