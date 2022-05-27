"""
Experimenting with uncertainty and the teacher
"""

# <codecell>
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
def run_incremental_vanilla(eps=0, goal_length=10, max_steps=1000):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps)
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


def run_incremental_unc(eps=0, certainty=0.75, goal_length=10, max_steps=1000):
    teacher = TeacherUncertainIncremental(certainty=certainty)
    env = UncertainCurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps)
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
traj_unc = run_incremental_unc(certainty=0.3)
traj_van = run_incremental_vanilla()

plt.plot(traj_unc)
plt.plot(traj_van)


# <codecell>
def _get_min_k(p_eps, certainty=0.95):
    raw_k = np.log(1 - certainty) / (-p_eps) - 1
    return int(np.floor(raw_k))

_get_min_k(-np.log(0.95), certainty=0.5)

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
