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


# TODO: clean up and work out rigorous tuning
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
        # self.prop_inc = goal_length // cut_factor
        self.prop_inc = 100
        self.inc = None
        self.transcript = []
        self.in_osc = False
    
    def next_action(self, state):
        curr_n, trans = state
        if self.in_osc:
            self.in_osc = False
            return int(curr_n + self.inc)
        else:
            self.transcript.extend(trans)

            if self.inc != None:   # incremental
                next_n = curr_n
                if len(self.transcript) > self.min_m:
                    if self.do_jump():
                        next_n = min(curr_n + self.inc, self.goal_length)
                        return int(next_n)
                    elif self.do_dive():
                        next_n //= self.cut_factor
                        self.inc = max(self.inc // 2, 1)
                        return int(next_n)

                if self.with_osc:
                    self.in_osc = True
                    return int(curr_n - self.inc)

                return int(next_n)

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

                return int(next_n)
            
            return int(self.prop_inc)
            
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


class TeacherExpAdaptive(Agent):
    def __init__(self, goal_length, prop_inc=100, discount=0.9):
        self.goal_length = goal_length
        self.inc = prop_inc
        self.discount = discount
        self.avgs = []
    
    def next_action(self, state):
        curr_n, trans = state
        self._consume_trans(trans)

        if len(self.avgs) == 1:
            return self.inc
        
        avg, last_avg = self.avgs[-1], self.avgs[-2]

        if avg > 0.8:
            if avg > last_avg:
                return min(curr_n + self.inc, self.goal_length)
            else:
                return max(curr_n - self.inc, 1)
        else:
            if avg > last_avg:
                return curr_n
            else:
                return max(curr_n - self.inc, 1)

        # TODO: need some measure of momentum / stability
        # inc = 10 * (avg - last_avg) * self.inc
        # return np.clip(curr_n + inc, 1, self.goal_length).astype(int)

        # if avg > last_avg:
        #     return min(curr_n + self.inc, self.goal_length)
        # elif avg < last_avg:
        #     return max(curr_n - self.inc, 1)
        # else:
        #     return curr_n
    
    def _consume_trans(self, trans):
        avg = self.avgs[-1] if len(self.avgs) > 0 else 0
        for x in trans:
            avg = (1 - self.discount) * x + self.discount * avg
        self.avgs.append(avg)

# class TeacherAdaptive(Agent):
#     def __init__(self, goal_length, prop_inc=100, tau=0.5, tau_low=0.1, conf=0.95, max_m_factor=3, abs_min_m=5) -> None:
#         super().__init__()
#         self.goal_length = goal_length
#         self.tau_low = tau_low
#         self.tau = tau
#         self.conf = conf

#         p_eps = -np.log(tau)
#         raw_min_m = int(np.round(np.log(1 - conf) / (-p_eps) - 1))
#         self.min_m = max(raw_min_m, abs_min_m)
#         self.max_m = self.min_m * max_m_factor

#         self.cut_factor = cut_factor
#         self.inc = prop_inc
#         self.transcript = []
#         self.in_osc = False
    
#     def next_action(self, state):
#         curr_n, trans = state
#         self.transcript.extend(trans)

#         if self.in_osc:
#             self.in_osc = False
#             return curr_n + self.inc
        
#         # TODO: incomplete


#     def do_jump(self, trans=None, thresh=None):
#         trans = self.transcript if trans == None else trans
#         for k in range(self.min_m, 1 + min(self.max_m, len(trans))):
#             prob_good = self._get_prob_good(trans[-k:], thresh=thresh)
#             if prob_good >= self.conf:
#                 return True

#         return False

#     def do_dive(self):
#         rev_trans = [not bit for bit in self.transcript]
#         return self.do_jump(rev_trans, 1 - self.threshold_low)

#     def _get_prob_good(self, transcript, thresh=None):
#         if thresh == None:
#             thresh = self.threshold

#         success = np.sum(transcript)
#         total = len(transcript)
#         prob_bad = beta.cdf(thresh, a=success+1, b=total-success+1)
#         return 1 - prob_bad


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

def run_adaptive(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, conf=0.2, tau=0.5, track_qs=False, **kwargs):
    teacher = TeacherAdaptive(goal_length, conf=conf, tau=tau, **kwargs)
    env = UncertainCurriculumEnv(goal_length=N, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr, 'n_step': 100}, anarchy_mode=True)
    traj = [env.N]
    env.reset()
    all_qs = []

    obs = (1, [])
    for _ in range(max_steps):
        action = teacher.next_action(obs)
        obs, _, is_done, _ = env.step(action)
        traj.append(env.N)

        all_qs.append(np.array([env.student.q_r[i] for i in range(goal_length)]))

        if is_done:
            break
    
    if track_qs:
        return traj, all_qs

    return traj

exp_teach = None
def run_adaptive_exp(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, track_qs=False, **kwargs):
    global exp_teach

    teacher = TeacherExpAdaptive(goal_length, **kwargs)
    env = UncertainCurriculumEnv(goal_length=N, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr, 'n_step': 100}, anarchy_mode=True)
    traj = [env.N]
    env.reset()
    all_qs = []

    obs = (1, [])
    for _ in range(max_steps):
        action = teacher.next_action(obs)
        obs, _, is_done, _ = env.step(action)
        traj.append(env.N)

        all_qs.append(np.array([env.student.q_r[i] for i in range(goal_length)]))

        if is_done:
            break
    
    if track_qs:
        exp_teach = teacher
        return traj, all_qs

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

        print('SCORE', env.student.score(goal_length))

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
max_steps = 1000
cut_factor = 1.5

N_eff = 10
eps_eff = 0

N, eps = to_cont(N_eff, eps_eff, dn_per_interval=100)

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs', 'qs'])

cases = [
    Case('Incremental (PK)', run_incremental_perfect, {'eps': eps, 'goal_length': N, 'lr': lr, 'threshold': 0.8}, [], []),
    # Case('Adaptive (BT)', run_adaptive, {'eps': eps, 'goal_length': N, 'lr': lr, 'threshold': 0.8, 'threshold_low': 0.2, 'tau':0.8, 'cut_factor': cut_factor, 'with_osc': False}, []),
    # Case('MCTS', run_mcts, {'eps_eff': eps_eff, 'goal_length': N_eff, 'lr': lr, 'gamma': 0.97, 'n_jobs': 48, 'n_iters': 100, 'pw_init': 10}, []),
    # Case('Adaptive (Osc)', run_adaptive, {'eps': eps, 'goal_length': N, 'lr': lr, 'threshold': 0.8, 'threshold_low': 0, 'tau':0.8, 'cut_factor': cut_factor, 'with_osc': True}, [], []),
    # Case('Adaptive (BT + Osc)', run_adaptive, {'eps': eps, 'goal_length': N, 'lr': lr, 'threshold': 0.8, 'threshold_low': 0.2, 'tau':0.8, 'cut_factor': cut_factor, 'with_osc': True}, []),
    Case('Adaptive (Exp)', run_adaptive_exp, {'eps': eps, 'goal_length': N, 'lr': lr, 'prop_inc': 100, 'discount': 0.9}, [], []),
]

for _ in tqdm(range(n_iters)):
    for case in cases:
        traj, qs = case.run_func(**case.run_params, max_steps=max_steps, T=T, track_qs=True)
        case.runs.append(traj)
        case.qs.append(qs)
        

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
### PLOT QS
fig, axs = plt.subplots(len(cases), n_iters)

for case, ax_set in zip(cases, axs):
    for qs, ax in zip(case.qs, ax_set):
        ax.imshow(np.array(qs).T, aspect='auto', vmin=0, vmax=10)

fig.tight_layout()

# <codecell>
### EXP TEACHER DEBUG
avgs = exp_teach.avgs
avgs_deriv = [0]

diffs = [x2 - x1 for x1, x2 in zip(avgs[:-1], avgs[1:])]
for d in diffs:
    a = avgs_deriv[-1]
    a = 0.05 * d + 0.95 * a
    avgs_deriv.append(a)


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