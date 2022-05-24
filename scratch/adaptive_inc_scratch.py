"""
Experimenting with adaptive incremental strategies
"""

# <codecell>
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

        if action <= self.max_step:
            d_length = action
        else:
            d_length = self.max_step - action

        self.N = np.clip(self.N + d_length, 1, self.goal_length)
        log_prob = self._get_score(self.N)  # TODO: update log-prob style in parent

        reward = 0
        is_done = False

        if self.N == self.goal_length and -log_prob < self.p_eps:
            reward = self.teacher_reward
            is_done = True
        
        return (self.N, log_prob), reward, is_done, {}


class TeacherAdaptiveIncremental(Agent):
    def __init__(self, target_threshold=0.5) -> None:
        super().__init__()
        self.log_targ_t = np.log(target_threshold)
        self.log_qe_est = None
    
    def next_action(self, state):
        _, log_prob = state
        return self._next_n(log_prob)
    
    def _next_n(self, log_prob):
        if self.log_qe_est == None:   # assume it's the first iteration
            self.log_qe_est = log_prob
            raw_n = self.log_targ_t / self.log_qe_est
        else:
            raw_n = (self.log_targ_t / self.log_qe_est) - (log_prob / self.log_qe_est)

        # print('---')
        # print('LOG_PROB', log_prob)
        # print('RAW_N', raw_n)
        return np.floor(np.max(raw_n, 0))   # TODO: try without np.max?
    
    def update(self, old_state, action, reward, next_state, is_done):
        pass

    def learn(self, *args, **kwargs):
        raise NotImplementedError('TeacherAdaptiveIncremental does not implement method `learn`')
    
    def reset(self):
        self.log_qe_est = None


def run_adaptive(eps=0, tau=0.45, goal_length=50, max_steps=1000):
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


def run_incremental(eps=0, goal_length=50, max_steps=1000):
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

# <codecell>
n_iters = 5
eps = -1
tau = 0.1
# tau = np.exp(-0.05) * 1/(1 + np.exp(-eps))

all_adp_runs = []
all_inc_runs = []

for _ in range(n_iters):
    all_adp_runs.append(run_adaptive(eps=eps, tau=tau))
    all_inc_runs.append(run_incremental(eps=eps))

# %%
label = {'label': 'Adaptive'}
for run in all_adp_runs:
    plt.plot(run, color='C0', **label)
    label = {}

label = {'label': 'Incremental'}
for run in all_inc_runs:
    plt.plot(run, color='C1', **label)
    label = {}

plt.legend()
plt.xlabel('Iteration')
plt.ylabel('N')
plt.title(f'Epsilon = {eps}')

plt.savefig(f'../fig/adp_v_inc_eps_{eps}.png')

# %%
