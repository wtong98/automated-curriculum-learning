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
    def __init__(self, goal_length, threshold=0.95, threshold_low=0.05, tau=0.5, conf=0.95, max_m_factor=3, abs_min_m=5, cut_factor=2, student=None, with_osc=False) -> None:
        super().__init__()
        self.goal_length = goal_length
        self.threshold = threshold
        self.threshold_low = threshold_low
        self.tau = tau
        self.conf = conf
        self.student = student
        self.with_osc = with_osc

        p_eps = -np.log(threshold)
        raw_min_m = int(np.round(np.log(1 - conf) / (-p_eps) - 1))
        self.min_m = max(raw_min_m, abs_min_m)
        self.max_m = self.min_m * max_m_factor

        self.cut_factor = cut_factor
        self.prop_inc = goal_length // cut_factor
        self.inc = None
        # self.inc = 100
        self.transcript = []
        self.in_osc = False
    
    # TODO: clean up and simplify
    def next_action(self, state):
        curr_n, trans = state
        if self.in_osc:
            self.in_osc = False
            return curr_n + self.inc
        else:
            self.transcript.extend(trans)

            if self.inc != None:   # incremental
                next_n = curr_n
                if len(self.transcript) > self.min_m:
                    if self.do_jump():
                        next_n = min(curr_n + self.inc, self.goal_length)
                        return next_n
                    elif self.do_dive():
                        next_n //= self.cut_factor
                        self.inc = max(self.inc // 2, 1)
                        return next_n

                if self.with_osc:
                    self.in_osc = True
                    return curr_n - self.inc

                return next_n

            elif len(self.transcript) > (self.min_m + self.max_m) // 2:   # binary search
                next_n = self.prop_inc

                if self.do_jump(thresh=self.tau):
                    self.inc = self.prop_inc
                    next_n = min(curr_n + self.inc, self.goal_length)
                    self.transcript = []
                else:
                    self.prop_inc //= self.cut_factor
                    next_n = self.prop_inc
                    self.transcript = []

                return next_n
            
            return self.prop_inc
            
    def do_jump(self, trans=None, thresh=None):
        trans = self.transcript if trans == None else trans
        for k in range(self.min_m, 1 + min(self.max_m, len(trans))):
            prob_good = self._get_prob_good(trans[-k:], thresh=thresh)
            if prob_good >= self.conf:
                return True

        return False

    def do_dive(self):
        rev_trans = [not bit for bit in self.transcript]
        return self.do_jump(rev_trans, 1 - self.threshold_low)

    def _get_prob_good(self, transcript, thresh=None):
        if thresh == None:
            thresh = self.threshold

        success = np.sum(transcript)
        total = len(transcript)
        prob_bad = beta.cdf(thresh, a=success+1, b=total-success+1)
        return 1 - prob_bad

# N, eps = to_cont(10, -1, 100)
# env = UncertainCurriculumEnv(goal_length=N, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=5, student_params={'lr': 0.1, 'n_step': 100}, anarchy_mode=True)
# traj = [env.N]
# env.reset()
# teacher = TeacherAdaptive(N, threshold=0.8, threshold_low=0.3, tau=0.8, student=env.student, with_osc=True)

# obs = (1, [])
# for _ in range(1000):
#     action = teacher.next_action(obs)
#     print('ACTION', action)
#     print('OSC', teacher.in_osc)
#     print('SCORE', teacher.student.score(env.N))
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

def run_incremental_opt(goal_length=3, eps=0, n_step=100, lr=0.1, tau=0.5, max_steps=1000):
    cum_prob = sig(eps)
    opt_inc = np.log(tau) / np.log(cum_prob)
    opt_inc = int(np.round(opt_inc))

    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr, 'n_step': n_step}, anarchy_mode=True)
    env.reset()

    curr_n = 1
    traj = [curr_n]

    score = env._get_score(1, train=False)
    for _ in range(max_steps):
        if -score < env.p_eps:
            curr_n = min(curr_n + opt_inc, goal_length)
        
        (_, score), _, is_done, _ = env.step(curr_n)
        traj.append(env.N)

        if is_done:
            break
    
    return traj

def run_incremental_perfect(goal_length=3, eps=0, T=5, n_step=100, lr=0.1, max_steps=1000, threshold=0.95, track_qs=False):
    env = CurriculumEnv(goal_length=goal_length, student_reward=20, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr, 'n_step': n_step}, anarchy_mode=True, track_qs=False)
    env.reset()
    traj = [env.N]
    all_qs = []

    for _ in range(max_steps):
        qs = np.array([eps + env.student.q_r[i] for i in range(goal_length)])
        if track_qs:
            all_qs.append(qs - eps)

        while len(qs) > 0 and np.exp(np.sum(np.log(sig(qs)))) < threshold:
            qs = qs[:-1]
        
        action = min(len(qs) + 1, goal_length)
        _, _, is_done, info = env.step(action)
        traj.append(env.N)

        if is_done:
            break
    
    if track_qs:
        return traj, all_qs
    else:
        return traj

# N, eps = to_cont(10, 0, dn_per_interval=100)
# traj = run_incremental_perfect(N, eps)

# traj = np.array(traj)
# diffs = traj[1:] - traj[:-1]

# fig, axs = plt.subplots(2, 1, figsize=(6, 8))
# axs[0].plot(traj)
# axs[0].set_title('PK trajectory')
# axs[1].plot(diffs)
# axs[1].set_title('PK increments')

# plt.savefig('../fig/pk_traj.png')

# N, eps = to_cont(10, 2, dn_per_interval=100)
# tau = 0.5

# cum_prob = sig(eps)
# opt_inc = np.log(tau) / np.log(cum_prob)
# opt_inc = int(np.round(opt_inc))

# env = CurriculumEnv(goal_length=N, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=5, student_params={'lr': 0.1, 'n_step': 100}, anarchy_mode=True)
# env.reset()

# curr_n = 1
# traj = [curr_n]
# all_scores = []

# score = env._get_score(1, train=False)
# for _ in range(100):
#     if -score < env.p_eps:
#         curr_n = min(curr_n + 100, N)
#         cum_prob *= np.exp(-env.p_eps)
#         opt_inc = np.log(tau) / np.log(cum_prob)
#         opt_inc = max(int(np.round(opt_inc)), 1)
#     all_scores.append(score)
    
#     (_, score), _, is_done, _ = env.step(curr_n)
#     traj.append(env.N)

#     if is_done:
#         break

# plt.plot(traj, alpha=0.8)
# plt.ylabel('N', color='C0')
# plt.gca().tick_params(axis='y', labelcolor='C0')
# ax = plt.gca().twinx()
# ax.plot(all_scores, color='C1', alpha=0.8)
# ax.set_ylabel('log(p)', color='C1')
# ax.tick_params(axis='y', labelcolor='C1')
# plt.title('Example N = 10 with Incremental')
# plt.savefig('example_n_10.png')



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

def run_adaptive(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, conf=0.2, tau=0.5, **kwargs):
    teacher = TeacherAdaptive(goal_length, conf=conf, tau=tau, **kwargs)
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

def run_mcts(eps_eff=0, goal_length=3, T=3, lr=0.1, max_steps=500, gamma=0.95, n_iters=500, n_jobs=4, pw_init=5):
    teacher = TeacherMctsCont(goal_length, n_jobs=n_jobs, n_iters=n_iters, pw_init=pw_init, gamma=gamma, student_params={'eps_eff': eps_eff})

    env = CurriculumEnv(goal_length=teacher.N, train_iter=999, train_round=T, p_eps=0.05, teacher_reward=10, student_reward=10, student_qe_dist=teacher.eps, student_params={'lr': lr, 'n_step':100}, anarchy_mode=True)
    traj = [env.N]
    env.reset()
    prev_qr = None
    prev_a = None

    for _ in range(max_steps):
        a = teacher.next_action(prev_a, prev_qr)
        print('TOOK ACTION', a)

        _, _, is_done, _ = env.step(a)
        traj.append(a)

        prev_a = a
        prev_qr = [env.student.q_r[i] for i in range(teacher.N)]

        if is_done:
            break

    print('done!')
    return traj

# <codecell>
n_iters = 3
T = 5
lr = 0.1
max_steps = 15000
cut_factor = 2

N_eff = 10
eps_eff = 0

N, eps = to_cont(N_eff, eps_eff, dn_per_interval=100)

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

cases = [
    Case('Incremental (PK)', run_incremental_perfect, {'eps': eps, 'goal_length': N, 'lr': lr}, []),
    # Case('Adaptive (BT)', run_adaptive, {'eps': eps, 'goal_length': N, 'lr': lr, 'threshold': 0.8, 'threshold_low': 0.2, 'tau':0.8, 'cut_factor': cut_factor, 'with_osc': False}, []),
    # Case('MCTS', run_mcts, {'eps_eff': eps_eff, 'goal_length': N_eff, 'lr': lr, 'gamma': 0.97, 'n_jobs': 48, 'n_iters': 50, 'pw_init': 10}, []),
    # Case('Adaptive (Osc)', run_adaptive, {'eps': eps, 'goal_length': N, 'lr': lr, 'threshold': 0.8, 'threshold_low': 0, 'tau':0.8, 'cut_factor': cut_factor, 'with_osc': True}, []),
    # Case('Adaptive (BT + Osc)', run_adaptive, {'eps': eps, 'goal_length': N, 'lr': lr, 'threshold': 0.8, 'threshold_low': 0.2, 'tau':0.8, 'cut_factor': cut_factor, 'with_osc': True}, []),
]

for _ in tqdm(range(n_iters)):
    for case in cases:
        case.runs.append(case.run_func(**case.run_params, max_steps=max_steps, T=T))

# <codecell>
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

ax2 = axs[0].twinx()
for i, case in enumerate(cases):
    label = {'label': case.name}
    for run in case.runs:
        axs[0].plot(run, color=f'C{i}', alpha=0.6, **label)
        label = {}

        # if i == 0:
        #     axs[0].plot(run, color=f'C{i}', alpha=0.7, **label)
        #     axs[0].set_yticks([0, 1,2,3])
        #     axs[0].set_ylabel('discrete', color=f'C{i}')
        #     axs[0].tick_params(axis='y', labelcolor=f'C{i}')
        # else:
        #     ax2.plot(run, color=f'C{i}', alpha=0.7, **label)
        #     ax2.set_ylabel('continuous', color=f'C{i}')
        #     ax2.tick_params(axis='y', labelcolor=f'C{i}')
        # label = {}

axs[0].legend()
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('N')
# axs[0].set_yticks(np.arange(N) + 1)

# axs[0].set_ylim((25, 50))
# axs[0].set_xlim((0, 30))


all_lens = [[len(run) for run in case.runs] for case in cases]
all_means = [np.mean(lens) for lens in all_lens]
all_serr = [2 * np.std(lens) / np.sqrt(n_iters) for lens in all_lens]
all_names = [case.name for case in cases]

axs[1].bar(np.arange(len(cases)), all_means, tick_label=all_names, yerr=all_serr)
axs[1].set_ylabel('Iterations')

fig.suptitle(f'Epsilon (eff) = {eps_eff}')
fig.tight_layout()
# plt.savefig(f'../fig/example_n_{N_eff}_eps_{eps_eff:.2f}.png')

# <codecell>
'''
# %% LONG COMPARISON PLOT
n_iters = 5
N_eff = 3
T = 5
lr = 0.1
max_steps = 1000
gamma = 0.9
conf = 0.2
bt_conf = 0.2
bt_tau = 0.05
eff_eps = np.arange(-2, 2.1, step=0.5)
# eff_eps = np.arange(-4, -1, step=0.5)

cont_params = [to_cont(N_eff, e) for e in eff_eps]
Ns, eps = zip(*cont_params)
N = Ns[0]

mc_iters = 1000
bins = 10

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

all_cases = [
    (
        # Case('Incremental (PK)', run_incremental_perfect, {'eps': e, 'goal_length': N, 'lr': lr}, []),
        Case('Adaptive (BT)', run_adaptive, {'eps': e, 'goal_length': N, 'lr': lr, 'threshold': 0.8, 'threshold_low': 0.2, 'tau':0.8, 'with_osc': False}, []),
        Case('MCTS', run_mcts, {'eps_eff': eff_eps[i], 'goal_length': N_eff, 'lr': lr}, []),
    ) for i, e in enumerate(eps)
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
offset = np.array([-1, 0, 1])
# offset = np.array([-1, 0])
x = np.arange(len(eps))
names = ['Incremental (PK)', 'Adaptive (BT)', 'MCTS']

# plt.yscale('log')

for name, off, mean, se in zip(names, width * offset, all_means, all_ses):
    plt.bar(x+off, mean, yerr=se, width=width, label=name)
    plt.xticks(x, labels=eff_eps)
    plt.xlabel('Epsilon')
    plt.ylabel('Iterations')

plt.legend()
plt.title(f'Teacher performance for N={N}')
plt.tight_layout()

plt.savefig(f'../fig/comparison_n_{N_eff}_cont.png')

# %%
## PLOT HEATMAP
T = 5
lr = 0.1
max_steps = 1500
cut_factor = 2

N_eff = 10
eps_effs = [-4, -3, -2, -1, 0, 1, 2]

results = [to_cont(N_eff, e, dn_per_interval=100) for e in eps_effs] 
Ns, eps = zip(*results)
N = Ns[0]

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

cases = [
    Case('Incremental (PK)', run_incremental_perfect, {'eps': e, 'goal_length': N, 'lr': lr}, [])
for e in eps]

for case in tqdm(cases):
    case.runs.append(case.run_func(**case.run_params, max_steps=max_steps, T=T, track_qs=True))
# %%
fig, axs = plt.subplots(2, 7, figsize=(21, 6))

for col, case in enumerate(cases):
    traj, qs = case.runs[0]
    qs = np.array(qs)

    axs[0,col].set_title(f'Eps (eff) = {eps_effs[col]}')
    axs[0,col].plot(traj)
    im = axs[1,col].imshow(qs.T, aspect='auto', vmin=0, vmax=20)

    fig.colorbar(im, ax=axs[1,col])

plt.savefig('../fig/heatmap_reward20.png')

# %%
'''