"""
Experimenting with adaptive incremental strategies
"""

# <codecell>
from collections import namedtuple
from curses import ALL_MOUSE_EVENTS
import gym
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../')
from env import *

# <codecell>
class MultistepCurriculumEnv(CurriculumEnv):
    def __init__(self, max_step=64, **kwargs):
        super().__init__(**kwargs)
        self.max_step = max_step
        self.action_space = gym.spaces.Discrete(2 * self.max_step + 1)  # wraps around to negatives

    def step(self, action):
        action = int(action)

        # TODO: anarchy mode _\m/
        d_length = action
        # if action <= self.max_step:
        #     d_length = action
        # else:
        #     d_length = self.max_step - action

        self.N = np.clip(self.N + d_length, 1, self.goal_length)
        log_prob = self._get_score(self.N)  # TODO: update log-prob style in parent

        reward = 0
        is_done = False

        if self.N == self.goal_length and -log_prob < self.p_eps:
            reward = self.teacher_reward
            is_done = True
        
        return (self.N, log_prob), reward, is_done, {}


class TeacherHastyIncremental(Agent):
    def __init__(self, target_threshold=0.5, with_perf_penalty=True) -> None:
        super().__init__()
        self.log_targ_t = np.log(target_threshold)
        self.with_perf_penalty = with_perf_penalty
        self.log_qe_est = None
    
    def next_action(self, state):
        _, log_prob = state
        return self._next_n(log_prob)
    
    def _next_n(self, log_prob):
        if self.log_qe_est == None:   # assume it's the first iteration
            self.log_qe_est = log_prob
            raw_n = self.log_targ_t / self.log_qe_est
        else:
            raw_n = (self.log_targ_t / self.log_qe_est) - self.with_perf_penalty * (log_prob / self.log_qe_est)

        # print('---')
        # print('LOG_PROB', log_prob)
        # print('RAW_N', raw_n)
        # print('RESULT', np.floor(raw_n, 0))
        return np.floor(raw_n)
    
    def update(self, old_state, action, reward, next_state, is_done):
        pass

    def learn(self, *args, **kwargs):
        raise NotImplementedError('TeacherHastyIncremental does not implement method `learn`')
    
    def reset(self):
        self.log_qe_est = None


class TeacherAdaptiveIncremental(Agent):
    def __init__(self, target_threshold=0.5, p_eps=0.05) -> None:
        super().__init__()
        self.log_targ_t = np.log(target_threshold)
        self.p_eps = p_eps
        self.log_qe_est = None
    
    def next_action(self, state):
        _, log_prob = state
        return self._next_n(log_prob)
    
    def _next_n(self, log_prob):
        if self.log_qe_est == None:
            self.log_qe_est = log_prob
            self.jump = np.floor(self.log_targ_t / self.log_qe_est + self.p_eps / self.log_qe_est)

            if self.jump < 1:
                print(f'warn: jump={self.jump} is too small, clipping to 1')
                self.jump = 1
        
        return self.jump if -log_prob < self.p_eps else 0
    
    def update(self, old_state, action, reward, next_state, is_done):
        pass

    def learn(self, *args, **kwargs):
        raise NotImplementedError('TeacherHastyIncremental does not implement method `learn`')
    
    def reset(self):
        self.log_qe_est = None


def run_hasty(eps=0, tau=0.45, with_perf_pen=True, goal_length=50, max_steps=5000):
    teacher = TeacherHastyIncremental(target_threshold=tau, with_perf_penalty=with_perf_pen)
    env = MultistepCurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps)
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


def run_adaptive(eps=0, tau=0.45, goal_length=50, max_steps=5000):
    teacher = TeacherAdaptiveIncremental(target_threshold=tau)
    env = MultistepCurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps)
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


def run_incremental(eps=0, goal_length=50, max_steps=5000):
    env = MultistepCurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps)
    env.reset()
    traj = [env.N]

    score = env._get_score(1, train=False)
    for _ in range(max_steps):
        if -score < env.p_eps:
            action = 1
        else:
            action = 0
        
        (_, score), _, is_done, _ = env.step(action)
        traj.append(env.N)

        if is_done:
            break
    
    return traj


def run_baseline(goal_length=50, max_steps=5000):
    env = MultistepCurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps)
    env.reset()
    traj = [env.N]

    score = env._get_score(1, train=False)
    for _ in range(max_steps):
        action = goal_length
        _, _, is_done, _ = env.step(action)
        traj.append(env.N)

        if is_done:
            break
    
    return traj


def sig(x):
    return 1 / (1 + np.exp(-x))

# <codecell>
n_iters = 5
max_steps = 5000
eps = 6.5
tau = np.exp(-0.05) * sig(eps)
tau = 0.5

Case = namedtuple('Case', ['name', 'run_func', 'run_params', 'runs'])

cases = [
    Case('Hasty', run_hasty, {'eps': eps, 'tau': tau}, []),
    Case('Adaptive', run_adaptive, {'eps': eps, 'tau': tau}, []),
    Case('Vanilla', run_incremental, {'eps': eps}, []),
    # Case('Baseline', run_baseline, {}, [])
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

all_lens = [[len(run) for run in case.runs] for case in cases]
all_means = [np.mean(lens) for lens in all_lens]
all_serr = [2 * np.std(lens) / np.sqrt(n_iters) for lens in all_lens]
all_names = [case.name for case in cases]

axs[1].bar(np.arange(len(cases)), all_means, tick_label=all_names, yerr=all_serr)
axs[1].set_ylabel('Iterations')

fig.suptitle(f'Epsilon = {eps}, Tau = {tau}')
fig.tight_layout()

# plt.savefig(f'../fig/eps_{eps}_tau_{tau:.2f}.png')

# %%

# %%
