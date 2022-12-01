"""
Visualizations for discrete methods

PLOTS TO MAKE
----------------
Benchmarks:
* Const N, vary eps (x3)
* Const N, vary eps, with generative noise model (x3, x3)
* Const eps, vary N (x3)

Illustrative examples:
* Trajectory plots
* Q-value-over-time plots
"""
# <codecell>
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.append('../')
from env import *
from experiment import *

save_path = Path('fig/')

# <codecell>
### BENCHMARK: Incremental vs. Matiisen
# TODO: tune Matiisen params

# iters = 5
# Ns = np.arange(3, 20, step=2)
# eps = np.linspace(-5, 3, num=9)

n_iters = 10
Ns = [3, 5, 10, 20]
eps = np.linspace(-1, 1, num=5)
# max_steps = 500

T = 3
lr = 0.1
alpha = 0.1
beta = 1
k = 5

all_cases = defaultdict(lambda: None)
raw_data = []

for N in tqdm(Ns):
    for e in eps:
        cases = [
            Case('Incremental', run_incremental, {'eps': e, 'goal_length': N}, []),
            Case('Online', run_online, {'eps': e, 'goal_length': N, 'alpha': alpha, 'beta': beta}, []),
            Case('Naive', run_naive, {'eps': e, 'goal_length': N, 'alpha': alpha, 'beta': beta, 'k': k}, []),
            Case('Window', run_window, {'eps': e, 'goal_length': N, 'alpha': alpha, 'beta': beta, 'k': k}, []),
            Case('Sampling', run_sampling, {'eps': e, 'goal_length': N, 'alpha': alpha, 'k': k}, []),
            Case('Random', run_random, {'eps': e, 'goal_length': N}, []),
            Case('Final', run_final_task_only, {'eps': e, 'goal_length': N}, []),
        ]

        run_exp(n_iters=n_iters, cases=cases, max_steps=N * 100, lr=lr, T=T)
        all_cases[(N, e)] = cases
        raw_data.extend(cases)

df = pd.DataFrame(raw_data)

# <codecell>
fig_dir = save_path / 'inc_v_matiisen'
if not fig_dir.exists():
    fig_dir.mkdir(parents=True)

def extract_plot_vals(row):
    traj_lens = [len(traj) for traj in row['runs']]

    return pd.Series([
        row['name'],
        row['run_params']['goal_length'],
        np.round(row['run_params']['eps'], decimals=2),
        traj_lens
    ], index=['name', 'N', 'eps', 'traj_lens'])


for N in Ns:
    plot_df = df.apply(extract_plot_vals, axis=1).explode('traj_lens')
    plot_df = plot_df.loc[plot_df['N'] == N]

    ax = sns.barplot(plot_df, x='eps', y='traj_lens', hue='name')
    ax.get_legend().set_title(None)
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Iterations')
    ax.set_title(f'N = {N}')
    sns.move_legend(ax, 'upper right')

    plt.savefig(fig_dir / f'N_{N}.svg')
    plt.clf()


# <codecell>

# <codecell>
### BENCHMARK: N x EPS
# iters = 5
# Ns = np.arange(3, 20, step=2)
# eps = np.linspace(-5, 3, num=9)

iters = 2
Ns = np.arange(3, 8, step=2)
eps = np.linspace(2, 2, num=4)

for N in Ns:
    for e in eps:
        pass


# %%
