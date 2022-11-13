"""
Finding optimal strategies for adaptive
"""

# <codecell>
import numpy as np

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


class TeacherTree:
    def __init__(self, splits, decisions=None, n_feats=2, n_splits=2) -> None:
        self.splits = splits.reshape(n_feats, n_splits - 1)
        if decisions == None:
            decisions = np.arange(n_feats * n_splits)

        self.decisions = decisions.reshape((n_splits,) * n_feats)
    
    def decide(self, feats):
        result = self.decisions
        for i, x in enumerate(feats):
            split = self.splits[i]
            dec_idx = np.sum(x > split)
            result = result[dec_idx]
        return result


# tree = TeacherTree(splits=np.array([0.1, 0.2, 0.3, 0.4]), decisions=np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']), n_feats=2, n_splits=3)
# tree.decide([0.3, 0.35])
class TeacherExpAdaptive(Agent):
    def __init__(self, goal_length, tree, prop_inc=100, discount=0.9):
        self.goal_length = goal_length
        self.tree = tree
        self.inc = prop_inc
        self.discount = discount
        self.avgs = []

    def dec_to_inc(self, dec, curr_n):
        if dec == 0:
            return max(curr_n - self.inc, 1)
        elif dec == 1:
            return curr_n
        elif dec == 2:
            return max(curr_n - self.inc, 1)
        elif dec == 3:
            return min(curr_n + self.inc, self.goal_length)
        
        return None
        
    
    def next_action(self, state):
        curr_n, trans = state
        self._consume_trans(trans)

        if len(self.avgs) == 1:
            return self.inc
        
        avg, last_avg = self.avgs[-1], self.avgs[-2]
        dec = self.tree.decide([avg, avg - last_avg])
        return self.dec_to_inc(dec, curr_n)
    
    def _consume_trans(self, trans):
        avg = self.avgs[-1] if len(self.avgs) > 0 else 0
        for x in trans:
            avg = (1 - self.discount) * x + self.discount * avg
        self.avgs.append(avg)

class TeacherExpAdaptiveOG(Agent):
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


# N_eff = 3
# eps_eff = 0
# T = 5
# lr = 0.1
# max_steps = 300

# N, eps = to_cont(N_eff, eps_eff, dn_per_interval=100)

# tree = TeacherTree(splits=np.array([0.8, 0]))

# teacher = TeacherExpAdaptiveOG(N)
# # teacher = TeacherExpAdaptive(N, tree)
# env = UncertainCurriculumEnv(goal_length=N, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr, 'n_step': 100}, anarchy_mode=True)
# traj = [env.N]
# env.reset()

# obs = (1, [])
# for _ in range(max_steps):
#     action = teacher.next_action(obs)
#     obs, _, is_done, _ = env.step(action)
#     traj.append(env.N)


#     if is_done:
#         break

# plt.plot(traj)


def run(splits, N_eff=10, eps_eff=0, n_iters=5, max_steps=300):
    N, eps = to_cont(N_eff, eps_eff, dn_per_interval=100)
    T = 5
    lr = 0.1
    results = []

    for _ in range(n_iters):
        tree = TeacherTree(splits)
        teacher = TeacherExpAdaptive(N, tree)

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

        results.append(len(traj))
    
    return results

r = run(np.array([0.8, 0]))
print(np.mean(r))