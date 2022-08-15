"""
Algorithms at the continuum limit
"""

# <codecell>
from collections import namedtuple
import matplotlib as plt
import numpy as np
from scipy.stats import beta

import sys
sys.path.append('../')

from env import *

class UncertainCurriculumEnv(CurriculumEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def step(self, action):
        (self.N, _), reward, is_done, info = super().step(action)
        return (self.N, info['transcript']), reward, is_done, {}

def to_cont(N=3, eps=0, dn_per_interval=100):
    prob = sig(eps) ** (1 / dn_per_interval)
    eps_cont = np.log(prob / (1 - prob))
    N_cont = N * dn_per_interval

    return N_cont, eps_cont


class TeacherUncertainOsc(Agent):
    def __init__(self, goal_length, tau=0.95, conf=0.75, max_m_factor=3, with_backtrack=False, bt_tau=0.05, bt_conf=0.75) -> None:
        super().__init__()
        self.goal_length = goal_length
        self.tau = tau
        self.conf = conf
        self.with_backtrack = with_backtrack
        self.bt_tau = bt_tau
        self.bt_conf = bt_conf

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
        elif self.with_backtrack and self.do_dive():
            self.n = max(self.n - 1, 1)
            next_n = self.n
        elif next_n == curr_n:
            next_n = max(self.n - 1, 1)
        
        return next_n
    
    def do_jump(self):
        trans = self.trans_dict[self.n]
        for k in range(self.min_m, 1 + min(self.max_m, len(trans))):
            prob_good = self._get_prob_good(trans[-k:])
            if prob_good >= self.conf:
                return True

        return False

    def do_dive(self):
        if self.n == 1:
            return

        trans = self.trans_dict[self.n - 1]
        rev_trans = [not bit for bit in trans]
        for k in range(self.min_m, 1 + min(self.max_m, len(trans))):
            prob_good = self._get_prob_good(rev_trans[-k:], 1 - self.bt_tau)
            if prob_good >= self.bt_conf:
                return True
        
        return False
    
    def _get_prob_good(self, transcript, tau=None):
        if tau == None:
            tau = self.tau

        success = np.sum(transcript)
        total = len(transcript)
        prob_bad = beta.cdf(tau, a=success+1, b=total-success+1)
        return 1 - prob_bad


class TeacherAdaptive(Agent):
    def __init__(self, goal_length, threshold=0.95, tau=0.5, conf=0.75, m_factor=2, abs_min_m=10, student=None) -> None:
        super().__init__()
        self.goal_length = goal_length
        self.threshold = threshold
        self.tau = tau
        self.conf = conf
        self.student = student

        self.trans_dict = defaultdict(list)
        self.n = 1

        p_eps = -np.log(threshold)
        raw_min_m = np.log(1 - conf) / (-p_eps) - 1
        self.m = max(int(np.floor(raw_min_m)) * m_factor, abs_min_m)

        self.interval_min = 1
        self.interval_max = goal_length
        self.interval_mid = int((self.interval_max + self.interval_min) / 2)
        self.inc = None
        # self.inc = 100
        self.transcript = []
    
    def next_action(self, state):
        curr_n, trans = state
        self.transcript.extend(trans)

        if self.inc != None:   # incremental
            next_n = curr_n
            if len(self.transcript) > self.m:
                prob_good = self._get_prob_good(self.transcript[-self.m:])
                # print('CURR TRANS', self.transcript[-self.m:])
                # print('PROB GOOD', prob_good)
                # print('ACTUAL SCORE', np.exp(self.student.score(curr_n)))
                if prob_good > self.conf:
                # if np.exp(self.student.score(curr_n)) > self.threshold:
                    next_n = min(curr_n + self.inc, self.goal_length)
                    self.transcript = []
            return next_n

        elif len(self.transcript) > self.m:   # binary search
            low, high = self._get_conf_intv(self.transcript)
            # print('TRANS', self.transcript)
            # print('INTV', (low, high))
            # print('LOW', self.interval_min)
            # print('MID', self.interval_mid)
            # print('HI', self.interval_max)

            if self.tau > high:
                self.interval_max = self.interval_mid
                self.transcript = []
            elif self.tau < low:
                self.interval_min = self.interval_mid
                self.transcript = []
            else:
                self.inc = self.interval_mid
            
            old_mid = self.interval_mid
            self.interval_mid = int((self.interval_max + self.interval_min) / 2)
            if old_mid == self.interval_mid:
                self.inc = self.interval_mid

        return self.interval_mid
            
    def _get_conf_intv(self, transcript, tau=None):
        if tau == None:
            tau = self.tau

        success = np.sum(transcript)
        total = len(transcript)
        return beta.interval(1 - self.conf, a=success+1, b=total-success+1)

    def _get_prob_good(self, transcript):
        success = np.sum(transcript)
        total = len(transcript)
        prob_bad = beta.cdf(self.threshold, a=success+1, b=total-success+1)
        return 1 - prob_bad

# N, eps = to_cont(3, -3, 100)
# env = UncertainCurriculumEnv(goal_length=N, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=5, student_params={'lr': 0.1, 'n_step': 100}, anarchy_mode=True)
# traj = [env.N]
# env.reset()
# teacher = TeacherAdaptive(N, conf=0.2, tau=0.5, student=env.student)

# obs = (1, [])
# for _ in range(1000):
#     action = teacher.next_action(obs)
#     print('ACTION', action)
#     obs, _, is_done, _ = env.step(action)
#     traj.append(env.N)

#     if is_done:
#         break

# plt.plot(traj)



def run_incremental(eps=0, goal_length=3, T=3, max_steps=1000, lr=0.1, n_step=1, inc=1):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr, 'n_step': n_step}, anarchy_mode=True)
    env.reset()

    curr_n = 1
    traj = [curr_n]

    score = env._get_score(1, train=False)
    for _ in range(max_steps):
        if -score < env.p_eps:
            curr_n = min(curr_n + inc, goal_length)
        
        (_, score), _, is_done, _ = env.step(curr_n)
        traj.append(env.N)

        if is_done:
            break
    
    return traj


def run_incremental_opt(eps=0, n_step=100, tau=0.5, **inc_kwargs):
    prob = sig(eps)
    opt_inc = np.log(tau) / np.log(prob)
    opt_inc = int(np.round(opt_inc))
    return run_incremental(inc=opt_inc, eps=eps, n_step=n_step, **inc_kwargs)


def run_osc(eps=0, confidence=0.75, goal_length=3, T=3, max_steps=1000, lr=0.1, **teacher_kwargs):
    teacher = TeacherUncertainOsc(goal_length, conf=confidence, **teacher_kwargs)
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

def run_adaptive(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, conf=0.2, tau=0.5):
    teacher = TeacherAdaptive(goal_length, conf=conf, tau=tau)
    env = UncertainCurriculumEnv(goal_length=N, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr, 'n_step': 100}, anarchy_mode=True)
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

# <codecell>
n_iters = 5
T = 5
lr = 0.01
max_steps = 10000

N_eff = 3
eps_eff = 0

N, eps = to_cont(N_eff, eps_eff, dn_per_interval=100)

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

cases = [
    # Case('Incremental', run_incremental, {'eps': eps, 'goal_length': N, 'lr': lr}, []),
    # Case('Incremental n-step', run_incremental, {'eps': eps, 'goal_length': N, 'lr': lr, 'n_step': 100}, []),
    # Case('Incremental n-step skip', run_incremental, {'eps': eps, 'goal_length': N, 'lr': lr, 'n_step': 100, 'inc': 22}, []),
    Case('Incremental n-step opt1', run_incremental_opt, {'eps': eps, 'goal_length': N, 'lr': lr, 'n_step': 100, 'tau':0.25}, []),
    Case('Incremental n-step opt2', run_incremental_opt, {'eps': eps, 'goal_length': N, 'lr': lr, 'n_step': 100, 'tau':0.5}, []),
    Case('Incremental n-step opt3', run_incremental_opt, {'eps': eps, 'goal_length': N, 'lr': lr, 'n_step': 100, 'tau':0.75}, []),
    Case('Adaptive', run_adaptive, {'eps': eps, 'goal_length': N, 'lr': lr, 'conf': 0.2, 'tau':0.5}, []),
    # Case('Incremental (w/ BT)', run_incremental_with_backtrack, {'eps': eps, 'goal_length': N, 'lr': lr}, []),
]

for _ in tqdm(range(n_iters)):
    for case in cases:
        case.runs.append(case.run_func(**case.run_params, max_steps=max_steps, T=T))

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
# axs[0].set_yticks(np.arange(N) + 1)

# axs[0].set_ylim((25, 50))
# axs[0].set_xlim((100, 200))


all_lens = [[len(run) for run in case.runs] for case in cases]
all_means = [np.mean(lens) for lens in all_lens]
all_serr = [2 * np.std(lens) / np.sqrt(n_iters) for lens in all_lens]
all_names = [case.name for case in cases]

axs[1].bar(np.arange(len(cases)), all_means, tick_label=all_names, yerr=all_serr)
axs[1].set_ylabel('Iterations')

fig.suptitle(f'Epsilon = {eps}')
fig.tight_layout()
# plt.savefig(f'../fig/osc_ex_n_{N}_eps_{eps}.png')

# %% LONG COMPARISON PLOT
n_iters = 5
N = 3
T = 5
lr = 0.1
max_steps = 500
gamma = 0.9
conf = 0.2
bt_conf = 0.2
bt_tau = 0.05
# eff_eps = np.arange(-2, 2.1, step=0.5)
eff_eps = np.arange(-4, -1, step=0.5)

cont_params = [to_cont(N, e) for e in eff_eps]
Ns, eps = zip(*cont_params)
N = Ns[0]

mc_iters = 1000
bins = 10

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

all_cases = [
    (
        # Case('Incremental', run_incremental, {'eps': e, 'goal_length': N, 'lr': lr}, []),
        Case('Incremental (n-step)', run_incremental_opt, {'eps': e, 'goal_length': N, 'lr': lr, 'n_step': 100, 'tau':0.5}, []),
        Case('Adaptive', run_adaptive, {'eps': e, 'goal_length': N, 'lr': lr, 'conf': 0.2, 'tau':0.5}, []),
    ) for e in eps
]

for _ in tqdm(range(n_iters)):
    for cases in all_cases:
        for case in cases:
            case.runs.append(case.run_func(**case.run_params, max_steps=max_steps, T=T))

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
        curr_ses.append(2 * np.std(run_lens) / np.sqrt(n_iters))

    all_means.append(curr_means)
    all_ses.append(curr_ses)


width = 0.2
# offset = np.array([-1, 0, 1])
offset = np.array([-1, 0])
x = np.arange(len(eps))
# names = ['Incremental', 'Incremental (n-step)', 'Adaptive']
names = ['Incremental (n-step)', 'Adaptive']

# plt.yscale('log')

for name, off, mean, se in zip(names, width * offset, all_means, all_ses):
    plt.bar(x+off, mean, yerr=se, width=width, label=name)
    plt.xticks(x, labels=eff_eps)
    plt.xlabel('Epsilon')
    plt.ylabel('Iterations')

plt.legend()
plt.title(f'Teacher performance for N={N}')
plt.tight_layout()

plt.savefig(f'../fig/osc_perf_n_{N}_low_eps_cont.png')
# %%

