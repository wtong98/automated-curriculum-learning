"""
Example implementations using differential evolution to identify an optimal
decision tree for the Adaptive teacher (with reduced search times), in discrete
and continuous sequence settings. Depending on the configuration, these search
procedures may take a great deal of time to run, so it is recommended to run
these on a compute cluster or a local machine you can leave alone for a long
period of time.

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from itertools import product
from multiprocessing import Pool

import numpy as np
from scipy.optimize import differential_evolution

import sys
sys.path.append('../')
from common import *
from env import *


def run_disc(splits, dec_to_idx):
    Ns = (10,)
    eps = (-2, -1, 0)
    n_iters = 5
    max_steps = 500

    T = 3
    lr = 0.1
    results = []

    for _ in range(n_iters):
        for N in Ns:
            for e in eps:
                tree = TeacherTree(splits, n_splits=2)
                teacher = TeacherExpAdaptive(N, tree, dec_to_idx, discrete=True)

                env = CurriculumEnv(goal_length=N, student_reward=10, student_qe_dist=e, train_round=T, student_params={'lr': lr, 'n_step': 1}, anarchy_mode=True, return_transcript=True)
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


def run_cont(splits, dec_to_idx):
    N_eff = 10
    eps_eff = 0
    n_iters = 5
    max_steps = 200

    N, eps = to_cont(N_eff, eps_eff, dn_per_interval=100)
    T = 3
    lr = 0.1
    results = []

    for _ in range(n_iters):
        tree = TeacherTree(splits)
        teacher = TeacherExpAdaptive(N, tree, dec_to_idx)

        env = CurriculumEnv(goal_length=N, student_reward=10, student_qe_dist=eps, train_round=T, student_params={'lr': lr, 'n_step': 100}, anarchy_mode=True, return_transcript=True)
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


def enumerate_maps(n_dec=4, n_idx=2):
    idxs = list(range(n_idx))
    for samp in product(idxs, repeat=n_dec):
        yield samp


# <codecell>
def cb(xk, **kwargs):
    print('XK', xk)

disc_init_args = [0, 1, 0, 2]

for _ in range(3):
    result = differential_evolution(run_disc, args=(disc_init_args,), bounds=[(0, 1), (-1, 1)], workers=-1, updating='deferred', maxiter=5, callback=cb)
    args = [(result.x, m) for m in enumerate_maps(n_dec=4, n_idx=3)]

    with Pool(16) as p:
        map_stats = p.starmap(run_disc, args)

    best_map = np.argmin(map_stats)
    dec_map = args[best_map][1]
    print('BEST', dec_map)

print('FINAL')
print('RESULT', result)
print('DEC MAP', dec_map)


# %%
cont_init_args = [3, 7, 0, 2]

for _ in range(3):
    result = differential_evolution(run_cont, args=(disc_init_args,), bounds=[(0, 1), (-1, 1)], workers=-1, updating='deferred', maxiter=5, callback=cb)
    args = [(result.x, m) for m in enumerate_maps(n_dec=4, n_idx=9)]

    with Pool(16) as p:
        map_stats = p.starmap(run_disc, args)

    best_map = np.argmin(map_stats)
    dec_map = args[best_map][1]
    print('BEST', dec_map)

print('FINAL')
print('RESULT', result)
print('DEC MAP', dec_map)