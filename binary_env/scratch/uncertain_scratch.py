"""
Experimenting with uncertainty and the teacher
"""

# <codecell>
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

import sys
sys.path.append('../')

from env import *

# <codecell>
class UncertainCurriculumEnv(CurriculumEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def step(self, action):
        action = int(action)
        d_length = action   # anarchy mode .\m/

        self.N = np.clip(self.N + d_length, 1, self.goal_length)
        trans = self._get_score(self.N)  # TODO: update log-prob style in parent

        reward = 0
        is_done = False

        log_prob = self.student.score(self.N)
        if self.N == self.goal_length and -log_prob < self.p_eps:
            reward = self.teacher_reward
            is_done = True
        
        return (self.N, trans), reward, is_done, {}

    def _get_score(self, length, train=True):
        trans = []

        def _update_trans(_, reward):
            result = int(reward > 0)
            trans.append(result)

        if train:
            self.student.learn(BinaryEnv(length, reward=self.student_reward), max_iters=self.train_iter, done_hook=_update_trans)

        return trans


class TeacherUncertainIncremental(Agent):
    def __init__(self, p_eps=0.05, certainty=0.75, max_k_factor=1.5) -> None:
        super().__init__()
        self.p_eps = p_eps
        self.success_prob = np.exp(-p_eps)
        self.certainty = certainty
        self.student_traj = []

        raw_min_k = np.log(1 - certainty) / (-p_eps) - 1
        self.min_k = int(np.floor(raw_min_k))
        self.max_k = int(self.min_k * max_k_factor)
    
    def next_action(self, state):
        _, trans = state
        self.student_traj.extend(trans)

        for k in range(self.min_k, self.max_k + 1):
            if k >= len(self.student_traj):
                return 0

            prob_good = self._get_prob_good(self.student_traj[-k:])
            if prob_good >= self.certainty:
                return 1

        return 0
    
    def _get_prob_good(self, transcript):
        success = np.sum(transcript)
        total = len(transcript)
        prob_bad = beta.cdf(self.success_prob, a=success+1, b=total-success+1)
        return 1 - prob_bad
    
    def update(self, old_state, action, reward, next_state, is_done):
        pass

    def learn(self, *args, **kwargs):
        raise NotImplementedError('TeacherHastyIncremental does not implement method `learn`')
    
    def reset(self):
        self.student_traj = []


# <codecell>
def run_incremental_vanilla(eps=0, goal_length=10, max_steps=1000, student_lr=0.005):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, student_params={'lr': student_lr})
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


def run_incremental_unc(eps=0, certainty=0.75, goal_length=10, max_steps=1000, student_lr=0.005):
    teacher = TeacherUncertainIncremental(certainty=certainty)
    env = UncertainCurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, student_params={'lr': student_lr})
    traj = [env.N]
    env.reset()

    obs = (1, env._get_score(1, train=False))
    for _ in range(max_steps):
        action = teacher.next_action(obs)
        obs, _, is_done, _ = env.step(action)
        traj.append(env.N)

        if is_done:
            break
    
    return traj


# TODO: make more formal plots
# traj_unc = run_incremental_unc(certainty=0.95)
# traj_van = run_incremental_vanilla()

# plt.plot(traj_unc)
# plt.plot(traj_van)

# <codecell>
n_iters = 10
# certainty = 0.95
max_steps = 5000

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

cases = [
    Case('Uncertain (lr=0.1)', run_incremental_unc, {'certainty': 0.95, 'student_lr': 0.1}, []),
    Case('Vanilla (lr=0.1)', run_incremental_vanilla, {'student_lr': 0.1}, []),
    Case('Uncertain (lr=0.02)', run_incremental_unc, {'certainty': 0.95, 'student_lr': 0.02}, []),
    Case('Vanilla (lr=0.02)', run_incremental_vanilla, {'student_lr': 0.02}, []),
    Case('Uncertain (lr=0.005)', run_incremental_unc, {'certainty': 0.95, 'student_lr': 0.005}, []),
    Case('Vanilla (lr=0.005)', run_incremental_vanilla, {'student_lr': 0.005}, []),
    Case('Uncertain (lr=0.001)', run_incremental_unc, {'certainty': 0.95, 'student_lr': 0.001}, []),
    Case('Vanilla (lr=0.001))', run_incremental_vanilla, {'student_lr': 0.001}, []),
]

for _ in tqdm(range(n_iters)):
    for case in cases:
        case.runs.append(case.run_func(**case.run_params, max_steps=max_steps))

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

for i, case in enumerate(cases):
    label = {'label': case.name}
    for run in case.runs:
        axs[0].plot(run, color=f'C{i}', alpha=0.7, **label)
        label = {}

axs[0].legend()
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('N')

all_lens = [[len(run) for run in case.runs] for case in cases]
all_means = [np.mean(lens) for lens in all_lens]
all_serr = [2 * np.std(lens) / np.sqrt(n_iters) for lens in all_lens]
all_names = [case.name for case in cases]

axs[1].bar(np.arange(len(cases)), all_means, tick_label=all_names, yerr=all_serr)
axs[1].set_ylabel('Iterations')

# fig.suptitle(f'Certainty = {certainty}')
fig.suptitle('Uncertain Teacher')
fig.tight_layout()

# plt.savefig('../fig/uncertain_teacher.png')

# <codecell>
## SPECIAL PLOT COMPARING VANILLA AND UNCERTAIN
unc_cases = cases[::2]
van_cases = cases[1::2]
diffs = []
errs = []
for unc, van, in zip(unc_cases, van_cases):
    unc_lens = [len(case) for case in unc.runs]
    van_lens = [len(case) for case in van.runs]

    unc_mean = np.mean(unc_lens)
    van_mean = np.mean(van_lens)
    unc_serr = np.std(unc_lens) / np.sqrt(n_iters)
    van_serr = np.std(van_lens) / np.sqrt(n_iters)

    diffs.append(np.abs(unc_mean - van_mean) / van_mean)
    errs.append(2 * np.sqrt(unc_serr ** 2 + van_serr ** 2) / van_mean)
    

plt.bar(range(len(diffs)), diffs, yerr=errs, tick_label=['0.1', '0.02', '0.005', '0.001'])
plt.xlabel('Learning rate')
plt.ylabel('Relative difference')
plt.savefig('../fig/diff.png')

# <codecell> SCRATCH WORK vvv
def _get_min_k(p_eps, certainty=0.95):
    raw_k = np.log(1 - certainty) / (-p_eps) - 1
    return int(np.floor(raw_k))

_get_min_k(-np.log(0.95), certainty=0.85)

# %%
def get_prob_good(transcript):
    success = np.sum(transcript)
    total = len(transcript)

    print(success)
    print(total)
    print(success / total)
    prob_bad = beta.cdf(0.8, a=success+1, b=total-success+1)
    return 1 - prob_bad

get_prob_good([1,0,0,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# %%
