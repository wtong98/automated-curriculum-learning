"""
Platform for tuning hyperparameters of various algorithms

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from dataclasses import dataclass, field
from multiprocessing import Pool

import numpy as np
import pandas as pd

from experiment import *

n_procs = 48

eps = [-2, -1, 0, 1, 2]
Ns = [3, 5, 10]
alphas = np.linspace(0, 1, num=20)
betas = 10 ** np.linspace(-1, 1, num=20)
ks = [1, 3, 5, 10, 20, 50]

# eps = [0, 1, 2]
# Ns = [3]
# alphas = np.linspace(0, 1, num=5)
# betas = 10 ** np.linspace(-1, 1, num=5)
# ks = [1, 3, 5]

def run(f, iters=3, **params):
    try:
        all_res = []
        for _ in range(iters):
            trajs = [f(eps=e, goal_length=N, max_steps=500, **params)[0] for e in eps for N in Ns]
            traj_lens = [len(t) for t in trajs]
            all_res.append(np.mean(traj_lens))
        return np.mean(all_res)
    except Exception as e:
        print('warn:', e)
        return np.inf

def proc_run_online(alpha, beta):
    return run(run_online, alpha=alpha, beta=beta)

def proc_run_naive(alpha, beta, k):
    return run(run_naive, alpha=alpha, beta=beta, k=k)

def proc_run_window(alpha, beta, k):
    return run(run_window, alpha=alpha, beta=beta, k=k)

def proc_run_sampling(alpha, k):
    return run(run_sampling, alpha=alpha, k=k)

@dataclass
class Case:
    f: Callable
    name: str
    params: list
    best_params: dict = field(default_factory=dict)
    best_score: int = 999

params = list([(a, b) for a in alphas for b in betas])
params_with_k = list([(a, b, k) for a in alphas for b in betas for k in ks])

if __name__ == '__main__':
    cases = [
        Case(proc_run_online, name='online', params=params),
        Case(proc_run_naive, name='naive', params=params_with_k),
        Case(proc_run_window, name='window', params=params_with_k),
        Case(proc_run_sampling, name='sampling', params=list([(a,k) for a in alphas for k in ks]))
    ]

    for case in cases:
        print('Running', case.name)
        with Pool(n_procs) as p:
            res = p.starmap(case.f, case.params)
            best_idx = np.argmin(res)
            best = case.params[best_idx]
            case.best_score = res[best_idx]

            case.best_params = {'alpha': best[0], 'beta': best[1]}
            if len(best) == 3:
                case.best_params['k'] = best[2]
    
    df = pd.DataFrame(cases)
    df.to_pickle('matiisen_params.pkl')
    print('done!')


# %%
