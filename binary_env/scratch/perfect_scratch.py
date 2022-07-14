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


teacher_cache = defaultdict(lambda: None)
def run_dp(eps=0, goal_length=3, bins=100, T=3, lr=0.1, max_steps=500):
    global teacher_cache
    if teacher_cache[eps] == None:
        teacher = TeacherPerfectKnowledgeDp(goal_length=goal_length, train_iters=T, n_bins_per_q=bins, student_params={'lr': lr, 'eps': eps})
        teacher.learn()
        teacher_cache[eps] = teacher
    else:
        teacher = teacher_cache[eps]

    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True)
    traj = [env.N]
    env.reset()

    N = goal_length
    qr = np.zeros(N)

    for _ in range(max_steps):
        a = teacher.next_action(qr)
        _, _, is_done, _ = env.step(a)
        traj.append(a)

        if is_done:
            break

        # print(f'action: {a}  state: {state}')
        qr = np.array([env.student.q_r[i] for i in range(N)])

    return traj


def run_mcts(n_iters=500, eps=0, goal_length=3, T=3, gamma=0.9, lr=0.01, max_steps=500):
    teacher = TeacherPerfectKnowledge(goal_length=goal_length, T=T, gamma=gamma, student_qe=eps, student_lr=lr, n_iters=n_iters)
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True)
    traj = [env.N]
    env.reset()

    N = goal_length
    qr = np.zeros(N)
    prev_a = None

    for _ in range(max_steps):
        a = teacher.next_action(prev_a, qr)
        _, _, is_done, _ = env.step(a)
        traj.append(a)

        if is_done:
            break

        # print(f'action: {a}  state: {state}')
        prev_a = a
        qr = np.array([env.student.q_r[i] for i in range(N)])

    return traj


def run_incremental(eps=0, goal_length=3, T=3, lr=0.01, max_steps=500):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr})
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


def run_incremental_with_backtrack(eps=0, goal_length=3, T=3, lr=0.01, max_steps=500):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True)
    env.reset()
    traj = [env.N]

    for _ in range(max_steps):
        qs = np.array([eps + env.student.q_r[i] for i in range(goal_length)])
        while len(qs) > 0 and -np.sum(np.log(sig(qs))) > env.p_eps:
            qs = qs[:-1]
        
        # print('---')
        # print('N', env.N)
        # print('QS', qs)
        # print('QS PROD', np.sum(np.log(sig(qs))))
        # print('Score', env._get_score(env.N, train=False))
        
        action = min(len(qs) + 1, goal_length)
        _, _, is_done, _ = env.step(action)
        traj.append(env.N)

        if is_done:
            break
    
    return traj
    

def sig(x): return 1 / (1 + np.exp(-x))

# <codecell>
n_iters = 1
N = 5
lr = 0.1
max_steps = 1000
bins = 10
eps = -2

mc_iters = 1000

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

cases = [
    # Case('Incremental', run_incremental, {'eps': eps, 'goal_length': N, 'lr': lr, 'T': 3}, []),
    Case('Incremental (w/ BT)', run_incremental_with_backtrack, {'eps': eps, 'goal_length': N, 'lr': lr}, []),
    # Case('MCTS', run_mcts, {'eps': eps, 'goal_length': N, 'lr': lr, 'n_iters': mc_iters}, []),
    # Case('DP', run_dp, {'eps': eps, 'goal_length': N, 'lr': lr, 'bins': bins}, []),
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
# plt.savefig(f'../fig/pk_n_{N}_eps_{eps}.png')


# %% LONG COMPARISON PLOT
n_iters = 10
N = 3
lr = 0.1
max_steps = 1000
gamma = 0.95
# eps = np.arange(-2, 2.1, step=0.5)
eps = np.arange(-4, 0.1, step=0.5)

mc_iters = 1000
bins = 10

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

all_cases = [
    (
        Case('Incremental', run_incremental, {'eps': e, 'goal_length': N, 'lr': lr}, []),
        Case('Incremental (w/ BT)', run_incremental_with_backtrack, {'eps': e, 'goal_length': N, 'lr': lr}, []),
        # Case('MCTS', run_mcts, {'eps': e, 'goal_length': N, 'lr': lr, 'n_iters': mc_iters, 'gamma': gamma}, []),
        Case('DP', run_dp, {'eps': e, 'goal_length': N, 'lr': lr, 'bins': bins}, []),
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
offset = np.array([-1, 0, 1])
# offset = np.array([-1, 0])
x = np.arange(len(eps))
# names = ['Incremental', 'MCTS', 'DP']
names = ['Incremental', 'Incremental (w/ BT)', 'DP']

for name, off, mean, se in zip(names, width * offset, all_means, all_ses):
    plt.bar(x+off, mean, yerr=se, width=width, label=name)
    plt.xticks(x, labels=eps)
    plt.xlabel('Epsilon')
    plt.ylabel('Iterations')

plt.legend()
plt.title(f'Teacher performance for N={N}')
plt.tight_layout()
plt.savefig(f'../fig/pk_performance_comparison_bins_{bins}_low_eps.png')


# <codecell>  INSPECT DP POLICY
# eps = np.arange(-2, 2.1, step=0.5)
# N = 3
# bins = 10

all_policies = []
# for e in tqdm(eps):
#     teacher = TeacherPerfectKnowledgeDp(goal_length=N, train_iters=3, n_bins_per_q=bins, student_params={'eps': e})
#     teacher.learn()
#     all_policies.append(teacher.policy)

for e in tqdm(eps):
    all_policies.append(teacher_cache[e].policy)

# <codecell>
slices = teacher_cache[0].state_axis

fig, axs = plt.subplots(len(eps), 2 * bins, figsize=(2 * bins * 3, len(eps) * 3))
mpb = None
mpb_e = None

prob_unif = np.ones(N) / N
ent_max = -np.sum(prob_unif * np.log(prob_unif))

for e, policy, ax_set in zip(eps, all_policies, axs):
    teacher = teacher_cache[e]
    for ax_p, ax_e, q3 in zip(ax_set[::2], ax_set[1::2], slices):
        img = np.zeros((bins, bins))
        ent = np.zeros((bins, bins))

        for q1_idx, q1 in enumerate(slices):
            for q2_idx, q2 in enumerate(slices):
                img[q1_idx, q2_idx] = policy[(q1, q2, q3)]
                vals = np.array([teacher._compute_value((q1, q2, q3), a) for a in [1, 2, 3]])
                probs = np.exp(vals) / (np.sum(np.exp(vals)))
                ent[q1_idx, q2_idx] = - np.sum(probs * np.log(probs))
                
        mpb = ax_p.imshow(img)
        ax_p.set_title(f'q3={q3}')
        ax_p.set_xlabel('q2')
        ax_p.set_xticks(slices[:-1:2])
        ax_p.set_xticklabels(list(slices[:-1:2]))
        ax_p.set_ylabel(f'q1, eps={e}')
        ax_p.set_yticks(slices[:-1:2])
        ax_p.set_yticklabels(list(slices[:-1:2]))

        mpb_e = ax_e.imshow(ent, vmin=0, vmax=ent_max)
        ax_e.set_title('entropy')

    fig.colorbar(mpb, ticks=np.arange(N) + 1, ax=ax_p)
    fig.colorbar(mpb_e, ax=ax_e)

fig.tight_layout()
plt.savefig(f'../fig/pk_policy_n_{N}_bins_{bins}_with_ent.png', dpi=150)


# <codecell>  OVERLAY WITH INCREMENTAL
slices = teacher_cache[0].state_axis
def sig(x): return 1 / (1 + np.exp(-x))

fig, axs = plt.subplots(len(eps), bins, figsize=(bins * 3, len(eps) * 3))
mpb = None

for e, policy, ax_set in zip(eps, all_policies, axs):
    teacher = teacher_cache[e]

    xx, yy = np.meshgrid(slices, slices)
    zz = np.zeros(xx.shape)

    for i, x in enumerate(slices):
        for j, y in enumerate(slices):
            if sig(x + e) * sig(y + e) > 0.95:
                zz[i, j] = 3
            elif sig(x + e) > 0.95:
                zz[i, j] = 2
            else:
                zz[i, j] = 1

    for ax, q3 in zip(ax_set, slices):
        img = np.zeros((bins, bins))

        for q1_idx, q1 in enumerate(slices):
            for q2_idx, q2 in enumerate(slices):
                img[q1_idx, q2_idx] = policy[(q1, q2, q3)]
                
        mpb = ax.imshow(img)
        ax.set_title(f'q3={q3}')
        ax.set_xlabel('q2')
        ax.set_xticks(slices[:-1:2])
        ax.set_xticklabels(list(slices[:-1:2]))
        ax.set_ylabel(f'q1, eps={e}')
        ax.set_yticks(slices[:-1:2])
        ax.set_yticklabels(list(slices[:-1:2]))

        ax.contour(xx, yy, zz, levels=1, colors='red')

    fig.colorbar(mpb, ticks=np.arange(N) + 1, ax=ax)

fig.tight_layout()
plt.savefig(f'../fig/pk_policy_n_{N}_bins_{bins}_with_incremental_overlay.png', dpi=150)



# %%
# teacher = teacher_cache[-1.5]
# slices = teacher.state_axis
# mpb = None

# fig, axs = plt.subplots(1, bins, figsize=(3 * bins, 3))

# for ax, q3 in zip(axs, slices):
#     img = np.zeros((bins, bins))
#     for q1_idx, q1 in enumerate(slices):
#         for q2_idx, q2 in enumerate(slices):
#             img[q1_idx, q2_idx] = teacher.policy[(q1, q2, q3)]
#     mpb = ax.imshow(img)
#     ax.set_title(f'q3={q3}')
#     ax.set_xlabel('q2')
#     ax.set_xticklabels([0] + list(slices))
#     ax.set_ylabel(f'q1, eps={e}')
#     ax.set_yticklabels([0] + list(slices))

# fig.colorbar(mpb, ticks=np.arange(N) + 1, ax=ax)
# fig.tight_layout()
# %%
eps = 0
tau = 0.95

# slices = np.linspace(0, 10, 100)
slices = teacher_cache[0].state_axis

xx, yy = np.meshgrid(slices, slices)
zz = np.zeros(xx.shape)

for i, x in enumerate(slices):
    for j, y in enumerate(slices):
        if sig(x + eps) * sig(y + eps) > tau:
            zz[i, j] = 3
        elif sig(x + eps) > tau:
            zz[i, j] = 2
        else:
            zz[i, j] = 1

# plt.imshow(zz)
plt.imshow(zz)
plt.contour(xx, yy, zz, levels=1, colors='red')
# %%
# TODO: try build oscillating teacher: (optimal under uncertainty?)
# oscillates until reaches threshold for confidence