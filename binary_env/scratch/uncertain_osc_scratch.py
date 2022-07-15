"""
Oscillating uncertain teacher
"""

# <codecell>
from collections import namedtuple
import matplotlib as plt
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
        (self.N, _), reward, is_done, info = super().step(action)
        return (self.N, info['transcript']), reward, is_done, {}


class TeacherUncertainOsc(Agent):
    def __init__(self, goal_length, tau=0.95, conf=0.75, max_m_factor=3) -> None:
        super().__init__()
        self.goal_length = goal_length
        self.tau = tau
        self.conf = conf

        self.trans_dict = defaultdict(list)
        self.n = 1

        p_eps = -np.log(tau)
        raw_min_m = np.log(1 - conf) / (-p_eps) - 1
        self.min_m = int(np.floor(raw_min_m))
        self.max_m = int(self.min_m * max_m_factor)

    def next_action(self, state):
        curr_n, trans = state
        self.trans_dict[curr_n].extend(trans)

        next_n = self.n
        if self.do_jump():
            self.n = min(self.n + 1, self.goal_length)
            next_n = self.n
        elif next_n == curr_n:
            next_n = max(self.n - 1, 1)
        
        return next_n
    
    # TODO: compute all probs to decide when to return
    def do_jump(self):
        trans = self.trans_dict[self.n]
        for k in range(self.min_m, 1 + min(self.max_m, len(trans))):
            prob_good = self._get_prob_good(trans[-k:])
            if prob_good >= self.conf:
                return True

        return False
    
    def _get_prob_good(self, transcript):
        success = np.sum(transcript)
        total = len(transcript)
        prob_bad = beta.cdf(self.tau, a=success+1, b=total-success+1)
        return 1 - prob_bad
        

def run_incremental(eps=0, goal_length=3, T=3, max_steps=1000, lr=0.1):
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


def run_osc(eps=0, confidence=0.75, goal_length=3, T=3, max_steps=1000, lr=0.1):
    teacher = TeacherUncertainOsc(goal_length, conf=confidence)
    env = UncertainCurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True)
    traj = [env.N]
    env.reset()

    obs = (1, [])
    for _ in range(max_steps):
        action = teacher.next_action(obs)
        obs, _, is_done, _ = env.step(action)
        traj.append(env.N)

        if is_done:
            break
    
    return traj


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

        qr = np.array([env.student.q_r[i] for i in range(N)])

    return traj

def run_incremental_with_backtrack(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True)
    env.reset()
    traj = [env.N]

    for _ in range(max_steps):
        qs = np.array([eps + env.student.q_r[i] for i in range(goal_length)])
        while len(qs) > 0 and -np.sum(np.log(sig(qs))) > env.p_eps:
            qs = qs[:-1]
        
        action = min(len(qs) + 1, goal_length)
        _, _, is_done, _ = env.step(action)
        traj.append(env.N)

        if is_done:
            break
    
    return traj


# <codecell>
n_iters = 5
N = 5
lr = 0.1
max_steps = 1000
bins = 10
eps = 0
conf=0.2

mc_iters = 1000

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

cases = [
    Case('Incremental', run_incremental, {'eps': eps, 'goal_length': N, 'lr': lr}, []),
    Case('Incremental (w/ BT)', run_incremental_with_backtrack, {'eps': eps, 'goal_length': N, 'lr': lr}, []),
    Case('Uncertain Osc', run_osc, {'eps': eps, 'goal_length': N, 'lr': lr, 'confidence': conf}, []),
    # Case('MCTS', run_mcts, {'eps': eps, 'goal_length': N, 'lr': lr, 'n_iters': mc_iters}, []),
    # Case('DP', run_dp, {'eps': eps, 'goal_length': N, 'lr': lr, 'bins': bins}, []),
]

for _ in tqdm(range(n_iters)):
    for case in cases:
        case.runs.append(case.run_func(**case.run_params, max_steps=max_steps))
# %%
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
max_steps = 2000
gamma = 0.95
conf = 0.9
# eps = np.arange(-2, 2.1, step=0.5)
eps = np.arange(-5, -1, step=0.5)

mc_iters = 1000
bins = 10

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

all_cases = [
    (
        # Case('Incremental', run_incremental, {'eps': e, 'goal_length': N, 'lr': lr}, []),
        Case('Incremental (w/ BT)', run_incremental_with_backtrack, {'eps': e, 'goal_length': N, 'lr': lr}, []),
        Case('Uncertain Osc', run_osc, {'eps': e, 'goal_length': N, 'lr': lr, 'confidence': conf}, []),
        # Case('MCTS', run_mcts, {'eps': e, 'goal_length': N, 'lr': lr, 'n_iters': mc_iters, 'gamma': gamma}, []),
        # Case('DP', run_dp, {'eps': e, 'goal_length': N, 'lr': lr, 'bins': bins}, []),
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
# offset = np.array([-1, 0, 1])
offset = np.array([-1, 0])
x = np.arange(len(eps))
# names = ['Incremental', 'Incremental (w/ BT)', 'Uncertain Osc']
names = ['Incremental (w/ BT)', 'Uncertain Osc']

for name, off, mean, se in zip(names, width * offset, all_means, all_ses):
    plt.bar(x+off, mean, yerr=se, width=width, label=name)
    plt.xticks(x, labels=eps)
    plt.xlabel('Epsilon')
    plt.ylabel('Iterations')

plt.legend()
plt.title(f'Teacher performance for N={N}')
plt.tight_layout()
plt.savefig(f'../fig/osc_perf_low_eps_conf_{conf}.png')
# %%
