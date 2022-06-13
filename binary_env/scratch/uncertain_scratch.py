"""
Experimenting with unconfidence and the teacher
"""

# <codecell>
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, linregress

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
    def __init__(self, p_eps=0.05, confidence=0.95, max_m_factor=3) -> None:
        super().__init__()
        self.p_eps = p_eps
        self.success_prob = np.exp(-p_eps)
        self.confidence = confidence
        self.student_traj = []

        raw_min_m = np.log(1 - confidence) / (-p_eps) - 1
        self.min_m = int(np.floor(raw_min_m))
        self.max_m = int(self.min_m * max_m_factor)
    
    def next_action(self, state):
        _, trans = state
        self.student_traj.extend(trans)

        for k in range(self.min_m, 1 + min(self.max_m, len(self.student_traj))):
            prob_good = self._get_prob_good(self.student_traj[-k:])
            if prob_good >= self.confidence:
                return 1

        return 0
    
    def _get_prob_good(self, transcript):
        success = np.sum(transcript)
        total = len(transcript)
        prob_bad = beta.cdf(self.success_prob, a=success+1, b=total-success+1)
        return 1 - prob_bad
    
    def reset(self):
        self.student_traj = []


class TeacherUncertainHasty(Agent):
    def __init__(self, target_threshold=0.5, confidence=0.95, max_k=10, max_m=100, n_particles=1000, eps_prior=None):
        self.target_threshold = target_threshold
        self.confidence = confidence
        self.max_k = max_k
        self.max_m = max_m
        self.n_particles=n_particles
        self.eps_prior = eps_prior or np.random.randn
        self.student_traj = []

    def next_action(self, state):
        _, history = state
        self.student_traj.extend(history)

        max_thresh = self._get_max_thresh()
        for k in range(1, self.max_k):
            conf = self._sim_prob(k, max_thresh)
            if conf < self.confidence:
                break
        
        return k - 1
    
    def _get_max_thresh(self):
        max_thresh = 1e-8
        n_success = 0
        n_fail = 0

        for result in self.student_traj[-self.max_m::][::-1]:
            n_success += result
            n_fail += not result
           
            thresh = beta.ppf(1 - self.confidence, n_success + 1, n_fail + 1)
            if thresh > max_thresh:
                max_thresh = thresh
        
        return max_thresh

    def _sim_prob(self, k, s_n):
        total_success = 0
        for _ in range(self.n_particles):
            samp_eps = self.eps_prior(k)
            prob = np.prod(1 / (1 + np.exp(-samp_eps)))
            if prob > self.target_threshold / s_n:
                total_success += 1

        return total_success / self.n_particles

    def reset(self):
        self.student_traj = []


class TeacherUncertainAdaptive(TeacherUncertainHasty):
    def __init__(self, p_eps=0.05, max_m_factor=3, **kwargs):
        super().__init__(**kwargs)
        self.p_eps = p_eps
        self.success_prob = np.exp(-p_eps)

        raw_min_m = np.log(1 - self.confidence) / (-p_eps) - 1
        self.min_m = int(np.floor(raw_min_m))
        self.max_m = int(self.min_m * max_m_factor)

    def next_action(self, state):
        _, trans = state
        self.student_traj.extend(trans)

        for m in range(self.min_m, 1 + min(self.max_m, len(self.student_traj))):
            prob_good = self._get_prob_good(self.student_traj[-m:])
            if prob_good >= self.confidence:
                jump = self.compute_jump()
                if jump == 0:
                    print('warn: jump=0, clipping to 1')
                    jump = 1
                return jump

        return 0
    
    def _get_prob_good(self, transcript):
        success = np.sum(transcript)
        total = len(transcript)
        prob_bad = beta.cdf(self.success_prob, a=success+1, b=total-success+1)
        return 1 - prob_bad
    
    def compute_jump(self):
        for k in range(1, self.max_k):
            conf = self._sim_prob(k, np.exp(-self.p_eps))
            if conf < self.confidence:
                break
        
        return k - 1

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


def run_incremental_unc(eps=0, confidence=0.95, goal_length=10, max_steps=1000, student_lr=0.005):
    teacher = TeacherUncertainIncremental(confidence=confidence)
    env = UncertainCurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, student_params={'lr': student_lr})
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


def run_hasty_unc(tau=0.4, goal_length=10, max_steps=1000, eps_prior_params=(0, 0.1), eps=0, student_lr=0.005):
    eps_prior = lambda k: np.random.normal(*eps_prior_params, size=k)

    teacher = TeacherUncertainHasty(target_threshold=tau, eps_prior=eps_prior)
    env = UncertainCurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, student_params={'lr': student_lr})
    traj = [env.N]
    env.reset()

    obs = (1, [])
    for _ in tqdm(range(max_steps)):
        action = teacher.next_action(obs)
        obs, _, is_done, _ = env.step(action)
        traj.append(env.N)

        if is_done:
            break

    return traj

    
def run_adp_unc(p_eps=0.05, tau=0.4, goal_length=10, max_steps=1000, eps_prior_params=(0, 0.1), eps=0, student_lr=0.005):
    eps_prior = lambda k: np.random.normal(*eps_prior_params, size=k)

    teacher = TeacherUncertainAdaptive(p_eps=p_eps, target_threshold=tau, eps_prior=eps_prior)
    env = UncertainCurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, student_params={'lr': student_lr})
    traj = [env.N]
    env.reset()

    obs = (1, [])
    for _ in tqdm(range(max_steps)):
        action = teacher.next_action(obs)
        obs, _, is_done, _ = env.step(action)
        traj.append(env.N)

        if is_done:
            break
    
    return traj


def sig(x):
    return 1 / (1 + np.exp(-x))

# traj_unc = run_incremental_unc()
# traj_hasty_unc = run_hasty_unc()
# traj_adp_unc = run_adp_unc()
# # traj_van = run_incremental_vanilla()

# plt.plot(traj_unc)
# # plt.plot(traj_van)
# plt.plot(traj_hasty_unc)
# plt.plot(traj_adp_unc)

# <codecell>  OBSERVE SCALING CURVES
ns = 3 * np.arange(1, 11)
n_iters = 5
lr = 0.01

max_steps = 5000
eps = 10
eps_prior_params = (eps, 0.1)
tau = 0.9

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

cases = defaultdict(list)
for n in ns:
    cases['Uncertain'].append(Case(str(n), run_incremental_unc, {'goal_length': n, 'eps': eps, 'student_lr': lr}, []))
    cases['Hasty'].append(Case(str(n), run_hasty_unc, {'goal_length': n, 'eps': eps, 'tau': tau, 'student_lr': lr, 'eps_prior_params': eps_prior_params}, []))
    cases['Adaptive'].append(Case(str(n), run_adp_unc, {'goal_length': n, 'eps': eps, 'tau': tau, 'student_lr': lr, 'eps_prior_params': eps_prior_params}, []))

for i in range(n_iters):
    print('Iter:', i)
    for name, cs in cases.items():
        print('Type:', name)
        for case in tqdm(cs):
            case.runs.append(case.run_func(**case.run_params, max_steps=max_steps))


# <codecell>
width = 0.5
offs = [-1, 0, 1]

all_xs = []
all_ys = []

for off, (name, cs) in zip(offs, cases.items()):
    run_lens = np.array([[len(run) for run in case.runs] for case in cs])
    mean = np.mean(run_lens, axis=1)
    std = np.std(run_lens, axis=1) / np.sqrt(n_iters)

    # plt.bar(ns + off * width, mean, width, yerr=2*std, label=name)
    plt.errorbar(ns, mean, fmt='o', yerr=2*std, label=name, alpha=0.7)
    all_xs.append(ns[:])
    all_ys.append(mean)

all_xs = np.concatenate(all_xs, axis=0)
all_ys = np.concatenate(all_ys, axis=0)

slope, intercept, r, _, _ = linregress(np.log(all_xs), np.log(all_ys))

plt.plot(ns, np.exp(slope * np.log(ns) + intercept), alpha=0.5)

plt.yscale('log')
plt.xscale('log')
plt.xticks(ns)
plt.legend()

plt.gcf().tight_layout()
plt.ylabel('Log Iterations')
plt.xlabel('N')
plt.savefig('../fig/scale_log_log.png')


# <codecell>
n_iters = 5
lr = 0.2

max_steps = 2500
eps = 1
eps_prior_params = (eps, 0.1)
tau = 0.65
N = 20

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

cases = [
    Case('Uncertain', run_incremental_unc, {'eps': eps, 'student_lr': lr, 'goal_length': N}, []),
    Case('Hasty', run_hasty_unc, {'eps': eps, 'tau': tau, 'student_lr': lr, 'eps_prior_params':eps_prior_params, 'goal_length': N}, []),
    # Case('Adaptive', run_adp_unc, {'eps': eps, 'tau': tau, 'student_lr': lr, 'eps_prior_params':eps_prior_params, 'goal_length': N}, []),
]

for _ in tqdm(range(n_iters)):
    for case in cases:
        case.runs.append(case.run_func(**case.run_params, max_steps=max_steps))

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

for i, case in enumerate(cases):
    label = {'label': case.name}
    for run in case.runs:
        axs[0].plot(run, color=f'C{i}', alpha=0.4, **label)
        label = {}

axs[0].legend()
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('N')
# axs[0].set_xlim((0, 300))

all_lens = [[len(run) for run in case.runs] for case in cases]
all_means = [np.mean(lens) for lens in all_lens]
all_serr = [2 * np.std(lens) / np.sqrt(n_iters) for lens in all_lens]
all_names = [case.name for case in cases]

axs[1].bar(np.arange(len(cases)), all_means, tick_label=all_names, yerr=all_serr)
axs[1].set_ylabel('Iterations')

# fig.suptitle(f'eps = {eps}   tau = {tau}')
fig.tight_layout()

plt.savefig(f'../fig/eps_{eps}_tau_{tau}.png')


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
def run_hasty_unc_adp(tau=0.4, goal_length=10, max_steps=1000, eps_prior_params=(0, 0.1), eps=0, student_lr=0.005):
    eps_prior = lambda k: np.random.normal(*eps_prior_params, size=k)

    teacher = TeacherUncertainHasty(target_threshold=tau, eps_prior=eps_prior)
    env = UncertainCurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, student_params={'lr': student_lr})
    traj = [env.N]
    env.reset()

    obs = (1, [])
    all_probs = []
    for _ in tqdm(range(max_steps)):
        action = teacher.next_action(obs)
        if action > 0 and env.N < goal_length:
            log_prob = env.student.score(env.N + action)
            all_probs.append(np.exp(log_prob))

        obs, _, is_done, _ = env.step(action)
        traj.append(env.N)

        if is_done:
            break
    
    return traj, teacher, env, all_probs


traj, all_probs = run_hasty_unc()
# %%
traj

# %%
