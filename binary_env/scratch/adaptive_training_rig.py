"""
Finding optimal strategies for adaptive
"""

# <codecell>
from itertools import product
from multiprocessing import Pool

import numpy as np
from scipy.optimize import differential_evolution

import sys
sys.path.append('../')
from env import *

# class UncertainCurriculumEnv(CurriculumEnv):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def step(self, action):
#         (self.N, _), reward, is_done, info = super().step(action)
#         return (self.N, info['transcript']), reward, is_done, {}

# def to_cont(N=3, eps=0, dn_per_interval=100):
#     prob = sig(eps) ** (1 / dn_per_interval)
#     eps_cont = np.log(prob / (1 - prob))
#     N_cont = N * dn_per_interval

#     return N_cont, eps_cont


# class TeacherTree:
#     def __init__(self, splits, decisions=None, n_feats=2, n_splits=2) -> None:
#         self.splits = splits.reshape(n_feats, n_splits - 1)
#         if decisions == None:
#             decisions = np.arange(n_feats * n_splits)

#         self.decisions = decisions.reshape((n_splits,) * n_feats)
    
#     def decide(self, feats):
#         result = self.decisions
#         for i, x in enumerate(feats):
#             split = self.splits[i]
#             dec_idx = np.sum(x > split)
#             result = result[dec_idx]
#         return result


# tree = TeacherTree(splits=np.array([0.4, 0.8, -0.1, 0.1]), decisions=np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']), n_feats=2, n_splits=3)
# tree.decide([0.3, 0.35])


# class TeacherExpAdaptive(Agent):
#     def __init__(self, goal_length, tree, dec_to_idx, discrete=False, prop_inc=100, shrink_factor=0.65, grow_factor=1.5, discount=0.8):
#         self.goal_length = goal_length
#         self.tree = tree
#         self.dec_to_idx = dec_to_idx
#         if discrete:
#             self.inc = 1
#             self.shrink_factor = 1
#             self.grow_factor = 1
#         else:
#             self.inc = prop_inc
#             self.shrink_factor = shrink_factor
#             self.grow_factor = grow_factor

#         self.discount = discount
#         self.avgs = []
    
#     def idx_to_act(self, idx):
#         inc_idx = idx // 3
#         jump_idx = idx % 3

#         if inc_idx == 1:
#             self.inc *= self.shrink_factor
#         elif inc_idx == 2:
#             self.inc *= self.grow_factor
            
#         if jump_idx == 0:
#             return -self.inc
#         elif jump_idx == 1:
#             return 0
#         elif jump_idx == 2:
#             return self.inc
        
#         return None

#     def dec_to_inc(self, dec, curr_n):
#         idx = self.dec_to_idx[dec]
#         act = self.idx_to_act(idx)
#         return np.clip(act + curr_n, 1, self.goal_length).astype(int)
    
#     def next_action(self, state):
#         curr_n, trans = state
#         self._consume_trans(trans)

#         if len(self.avgs) == 1:
#             return self.inc
        
#         avg, last_avg = self.avgs[-1], self.avgs[-2]
#         dec = self.tree.decide([avg, avg - last_avg])
#         return self.dec_to_inc(dec, curr_n)
    
#     def _consume_trans(self, trans):
#         avg = self.avgs[-1] if len(self.avgs) > 0 else 0
#         for x in trans:
#             avg = (1 - self.discount) * x + self.discount * avg
#         self.avgs.append(avg)


def run_cont(splits=None, dec_to_idx=None, N_eff=10, eps_eff=0, n_iters=5, max_steps=200):
    if splits == None or dec_to_idx == None:   # load optimal values
        splits = np.array([0.7, 0])
        dec_to_idx = np.array([3, 7, 0, 2])

    N, eps = to_cont(N_eff, eps_eff, dn_per_interval=100)
    T = 5
    lr = 0.1
    results = []

    for _ in range(n_iters):
        tree = TeacherTree(splits)
        teacher = TeacherExpAdaptive(N, tree, dec_to_idx)

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
    
    return np.mean(results)


def run_disc_split3(splits, *args):
    Ns = (10,)
    eps = (-2, -1, 0)
    max_steps = 500

    assert len(args) == 1
    dec_to_idx = args[0]

    T = 5
    lr = 0.1
    results = []

    for _ in range(5):
        for N in Ns:
            for e in eps:
                tree = TeacherTree(splits, n_splits=3)
                teacher = TeacherExpAdaptive(N, tree, dec_to_idx, discrete=True)

                env = CurriculumEnv(goal_length=N, student_reward=10, student_qe_dist=e, train_iter=999, train_round=T, student_params={'lr': lr, 'n_step': 100}, anarchy_mode=True, return_transcript=True)
                traj = [env.N]
                env.reset()

                obs = (1, [])
                for _ in range(max_steps):
                    action = teacher.next_action(obs)
                    obs, _, is_done, _ = env.step(action)
                    traj.append(env.N)

                    if is_done:
                        break

                score = len(traj)
                if score >= max_steps:
                    score = np.inf
                results.append(score)
    
    return np.mean(results)



def enumerate_maps(n_dec=9, n_idx=3):
    idxs = list(range(n_idx))
    for samp in product(idxs, repeat=n_dec):
        yield samp

split3_init_args = [0, 1, 1, 0, 1, 1, 0, 2, 2]
run_disc_split3([0.4, 0.8, -0.1, 0.1], [0, 1, 1, 0, 1, 1, 0, 2, 2])

# <codecell>
# def cb(xk, **kwargs):
#     print('XK', xk)

# result = differential_evolution(run_disc_split3, args=(split3_init_args,), bounds=[(0, 1), (0, 1), (-1, 1), (-1, 1)], workers=-1, updating='deferred', maxiter=10, callback=cb)

# <codecell>
def cb(xk, **kwargs):
    print('XK', xk)

dec_map = [0, 1, 0, 2]

# TODO: implement coordinated, greedy ascent across varying N's and eps's
for _ in range(3):
    result = differential_evolution(run_disc_split3, args=(split3_init_args,), bounds=[(0, 1), (0, 1), (-1, 1), (-1, 1)], workers=-1, updating='deferred', maxiter=5, callback=cb)
    args = [(result.x, m) for m in enumerate_maps(n_dec=9, n_idx=3)]

    with Pool(16) as p:
        map_stats = p.starmap(run_disc_split3, args)

    best_map = np.argmin(map_stats)
    dec_map = args[best_map][1]
    print('BEST', dec_map)

print('FINAL')
print('RESULT', result)
print('DEC MAP', dec_map)

''' TODO: tune and redo with new discount
FINAL RESULTS

RESULT      fun: 133.8
 message: 'Maximum number of iterations has been exceeded.'
    nfev: 318
     nit: 5
 success: False
       x: array([6.91197986e-01, 6.26838075e-04])
DEC MAP [3, 7, 0, 2]
'''

