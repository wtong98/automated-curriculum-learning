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
from viz.experiment import run_adp_exp_disc

class UncertainCurriculumEnv(CurriculumEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def step(self, action):
        (self.N, _), reward, is_done, info = super().step(action)
        return (self.N, info['transcript']), reward, is_done, {}


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


class TeacherExpAdaptive(Agent):
    def __init__(self, goal_length, discount=0.9):
        self.goal_length = goal_length
        self.discount = discount
        self.avgs = []
    
    def next_action(self, state):
        curr_n, trans = state
        self._consume_trans(trans)

        if len(self.avgs) == 1:
            return 1
        
        avg, last_avg = self.avgs[-1], self.avgs[-2]

        if avg > 0.8:
            if avg > last_avg:
                return min(curr_n + 1, self.goal_length)
            else:
                return max(curr_n - 1, 1)
        else:
            if avg > last_avg:
                return curr_n
            else:
                return max(curr_n - 1, 1)
    
    def _consume_trans(self, trans):
        avg = self.avgs[-1] if len(self.avgs) > 0 else 0
        for x in trans:
            avg = (1 - self.discount) * x + self.discount * avg
        self.avgs.append(avg)

# TODO: implement for trail teacher
class TeacherDoubleExpAdaptive(Agent):
    def __init__(self, goal_length, data_discount=0.5, trend_discount=0.2):
        self.goal_length = goal_length
        self.data_discount = data_discount
        self.trend_discount = trend_discount

        self.data_hist = []
        self.trend_hist = []
    
    def next_action(self, state):
        curr_n, trans = state
        self._consume_trans(trans)

        if len(self.data_hist) == 1:
            return 1
        
        data_avg = self.data_hist[-1]
        trend_avg = self.trend_hist[-1]

        if data_avg > 0.8:
            if trend_avg > 0:
                return min(curr_n + 1, self.goal_length)
            else:
                return max(curr_n - 1, 1)
        else:
            if trend_avg > 0:
                return curr_n
            else:
                return max(curr_n - 1, 1)
    
    def _consume_trans(self, trans):
        if len(self.data_hist) == 0:
            self.data_hist.append(0)
            self.trend_hist.append(0)
            return
        
        last_data_avg = self.data_hist[-1]
        trend_avg = self.trend_hist[-1]

        for x in trans:
            data_avg = (1 - self.data_discount) * x + self.data_discount * (last_data_avg + trend_avg)
            trend_avg = (1 - self.trend_discount) * (data_avg - last_data_avg) + self.trend_discount * trend_avg
            last_data_avg = data_avg

        self.data_hist.append(data_avg)
        self.trend_hist.append(trend_avg)


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


def run_incremental_with_partial_bt(eps=0, goal_length=3, T=3, max_steps=1000, lr=0.1):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr})
    env.reset()
    traj = [env.N]

    score = env._get_score(1, train=False)
    for _ in range(max_steps):
        if -score < env.p_eps:
            action = 2
        elif np.exp(score) < 1 - np.exp(-env.p_eps):
            action = 0
        else:
            action = 1
        
        (_, score), _, is_done, _ = env.step(action)
        traj.append(env.N)

        if is_done:
            break
    
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


def run_adp_exp(eps=0, goal_length=3, T=3, max_steps=1000, lr=0.1, **teacher_kwargs):
    teacher = TeacherExpAdaptive(goal_length, **teacher_kwargs)
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


de_teacher = None
def run_adp_double_exp(eps=0, goal_length=3, T=3, max_steps=1000, lr=0.1, **teacher_kwargs):
    global de_teacher

    teacher = TeacherDoubleExpAdaptive(goal_length, **teacher_kwargs)
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
    
    de_teacher = teacher
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

agent = None
def run_pomcp(n_iters=5000, eps=0, goal_length=3, T=3, gamma=0.9, lr=0.1, max_steps=500):
    global agent

    agent = TeacherPomcpAgentClean(goal_length=N, 
                            T=T, bins=10, p_eps=0.05, gamma=gamma, 
                            n_particles=n_iters, q_reinv_prob=0.25)
    env = CurriculumEnv(goal_length=goal_length, train_iter=999, train_round=T, p_eps=0.05, teacher_reward=10, student_reward=10, student_qe_dist=eps, student_params={'lr': lr})
    traj = [env.N]
    prev_obs = env.reset()
    prev_a = None

    for _ in range(max_steps):
        a = agent.next_action(prev_a, prev_obs)

        state, _, is_done, _ = env.step(a)
        traj.append(env.N)

        obs = agent._to_bin(state[1])
        prev_a = a
        prev_obs = obs

        if is_done:
            break
    
    return traj


def run_pomcp_old(n_iters=5000, eps=0, goal_length=3, T=3, gamma=0.9, lr=0.1, max_steps=500):
    agent = TeacherPomcpAgent(goal_length=N, 
                            lookahead_cap=1, 
                            T=T, bins=10, p_eps=0.05, gamma=gamma, 
                            n_particles=n_iters, q_reinv_scale=3, q_reinv_prob=0.25)
    env = CurriculumEnv(goal_length=goal_length, train_iter=999, train_round=T, p_eps=0.05, teacher_reward=10, student_reward=10, student_qe_dist=eps, student_params={'lr': lr})
    traj = [env.N]
    prev_obs = env.reset()
    prev_a = None

    for _ in range(max_steps):
        a = agent.next_action(prev_a, prev_obs)

        state, _, is_done, _ = env.step(a)
        traj.append(a)

        obs = agent._to_bin(state[1])
        prev_a = a
        prev_obs = obs

        if is_done:
            break
    
    del agent.tree
    del agent
    return traj


def run_pomcp_with_retry(max_retries=5, max_steps=500, **kwargs):
    for i in range(max_retries):
        try:
            return run_pomcp(max_steps=max_steps, **kwargs)
        except Exception as e:
            print('pomcp failure', i+1)
            print(e)
    
    return [0] * max_steps
    

# <codecell>
n_iters = 100
T = 3
N = 10
lr = 0.1
max_steps = 500
bins = 10
eps = -1
conf=0.2

mc_iters = 1000

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

cases = [
    # Case('Incremental', run_incremental, {'eps': eps, 'goal_length': N, 'lr': lr}, []),
    # Case('Incremental (w/ BT)', run_incremental_with_backtrack, {'eps': eps, 'goal_length': N, 'lr': lr}, []),
    # Case('Incremental (w/ PBT)', run_incremental_with_partial_bt, {'eps': eps, 'goal_length': N, 'lr': lr}, []),
    # Case('Uncertain Osc', run_osc, {'eps': eps, 'goal_length': N, 'lr': lr, 'confidence': conf}, []),
    # Case('Uncertain Osc (w/ BT)', run_osc, {'eps': eps, 'goal_length': N, 'lr': lr, 'confidence': conf, 'with_backtrack': True, 'bt_conf': conf, 'bt_tau': 0.25}, []),
    # Case('Oscillator', run_osc, {'eps': eps, 'goal_length': N, 'lr': lr, 'confidence': conf, 'with_backtrack': True, 'bt_conf': conf, 'bt_tau': 0.25}, []),
    # Case('Oscillator (95)', run_osc, {'eps': eps, 'goal_length': N, 'lr': lr, 'confidence': 0.95, 'with_backtrack': True, 'bt_conf': 0.95, 'tau': 0.1, 'bt_tau': 0.02}, []),
    Case('Adaptive (Exp)', run_adp_exp, {'eps': eps, 'goal_length': N, 'lr': lr, 'discount': 0.8}, []),
    Case('Adaptive (Exp x2)', run_adp_double_exp, {'eps': eps, 'goal_length': N, 'lr': lr, 'data_discount': 0.8, 'trend_discount': 0.9}, []),
    # Case('POMCP', run_pomcp, {'eps': eps, 'goal_length': N, 'lr': lr}, []),
    # Case('POMCP (old)', run_pomcp_old, {'eps': eps, 'goal_length': N, 'lr': lr}, []),
    # Case('MCTS', run_mcts, {'eps': eps, 'goal_length': N, 'lr': lr, 'n_iters': mc_iters}, []),
    # Case('DP', run_dp, {'eps': eps, 'goal_length': N, 'lr': lr, 'bins': bins}, []),
]

for _ in tqdm(range(n_iters)):
    for case in cases:
        case.runs.append(case.run_func(**case.run_params, max_steps=max_steps, T=T))
# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

for i, case in enumerate(cases):
    label = {'label': case.name}
    for run in case.runs:
        axs[0].plot(np.arange(len(run)) + i * 0.1, run, color=f'C{i}', alpha=0.8, linewidth=1, **label)
        label = {}

axs[0].legend()
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('N')
axs[0].set_yticks([1, 2, 3])

all_lens = [[len(run) for run in case.runs] for case in cases]
all_means = [np.mean(lens) for lens in all_lens]
all_serr = [2 * np.std(lens) / np.sqrt(n_iters) for lens in all_lens]
all_names = [case.name for case in cases]

axs[1].bar(np.arange(len(cases)), all_means, tick_label=all_names, yerr=all_serr)
axs[1].set_ylabel('Iterations')

fig.suptitle(f'Epsilon = {eps}')
fig.tight_layout()

# axs[0].set_xlim((0, 100))

# plt.savefig(f'../fig/osc_ex_n_{N}_eps_{eps}.png')


# %% LONG COMPARISON PLOT
n_iters = 5
N = 3
T = 5
lr = 0.1
max_steps = 1000
gamma = 0.9
conf = 0.2
bt_conf = 0.2
bt_tau = 0.05
eps = np.arange(-3, 2.1, step=0.5)
# eps = np.arange(-4, -1, step=0.5)
# eps = np.arange(-7, -3, step=0.5)

mc_iters = 1000
bins = 10

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

all_cases = [
    (
        Case('Incremental', run_incremental, {'eps': e, 'goal_length': N, 'lr': lr}, []),
        # Case('Incremental (w/ BT)', run_incremental_with_backtrack, {'eps': e, 'goal_length': N, 'lr': lr}, []),
        # Case('Incremental (w/ PBT)', run_incremental_with_partial_bt, {'eps': e, 'goal_length': N, 'lr': lr}, []),
        # Case('Uncertain Osc', run_osc, {'eps': e, 'goal_length': N, 'lr': lr, 'confidence': conf}, []),
        # Case('Uncertain Osc (w/ BT)', run_osc, {'eps': e, 'goal_length': N, 'lr': lr, 'confidence': conf, 'with_backtrack': True, 'bt_conf': bt_conf, 'bt_tau': bt_tau}, []),
        Case('Oscillator', run_osc, {'eps': e, 'goal_length': N, 'lr': lr, 'confidence': conf, 'with_backtrack': True, 'bt_conf': bt_conf, 'bt_tau': bt_tau}, []),
        # Case('POMCP', run_pomcp_with_retry, {'eps': e, 'goal_length': N, 'lr': lr, 'gamma': gamma}, []),
        # Case('MCTS', run_mcts, {'eps': e, 'goal_length': N, 'lr': lr, 'n_iters': mc_iters, 'gamma': gamma}, []),
        # Case('DP', run_dp, {'eps': e, 'goal_length': N, 'lr': lr, 'bins': bins}, []),
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


width = 0.27
# offset = np.array([-2, -1, 0, 1, 2])
# offset = np.array([-1, 0, 1])
offset = np.array([-1, 0])
x = np.arange(len(eps))
names = ['Incremental', 'Oscillator']
# names = ['Incremental', 'Incremental (w/ BT)', 'Incremental (w/ PBT)', 'Osc', 'Osc (w/ BT)']
# names = ['Incremental (w/ BT)', 'Incremental (w/ PBT)', 'Osc (w/ BT)']
# names = ['Incremental', 'Incremental (w/ BT)', 'Osc', 'Osc (w/ BT)', 'POMCP']
# names = ['Incremental (w/ BT)', 'Osc (w/ BT)', 'POMCP']
# names = ['Incremental (w/ BT)', 'Osc', 'Osc (w/ BT)']
# names = ['Incremental (w/ BT)', 'Uncertain Osc']
# names = ['Incremental (w/ BT)', 'Uncertain Osc (w/ BT)']

# plt.yscale('log')
plt.gcf().set_size_inches(4, 2.8)

for name, off, mean, se in zip(names, width * offset, all_means, all_ses):
    plt.bar(x+off, mean, yerr=se, width=width, label=name)
    plt.xticks(x, labels=eps)
    plt.xlabel('Epsilon')
    plt.ylabel('Iterations')

plt.yscale('log')
plt.legend()

# plt.title(f'Teacher performance for N={N}')
plt.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.15)

plt.savefig(f'../fig/osc_perf_n_{N}.svg')
# %%
