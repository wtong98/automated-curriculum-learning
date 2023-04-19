"""
Continuous environment figures
"""

# <codecell>
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import *

# <codecell>
### INCREMENTAL CONJOINED
fig, axs = plt.subplots(3, 2, figsize=(9, 8))

N = 10
eps_s = [2, 0, -2]

for i, eps in enumerate(eps_s):
    traj, info = run_inc_cont(eps=eps, goal_length=N)
    plot_traj_and_qr(traj, info['qr'], eps, N, ax=axs[i,0], n_step=100)
    plot_traj_slices(info['qr'], axs[i,1], eps, n_steps=100)

plt.savefig('fig/inc_cont_conjoined.png')
# %%
### ADAPTIVE CONJOINED
fig, axs = plt.subplots(3, 2, figsize=(9, 8))

N = 10
eps_s = [2, 0, -2]

for i, eps in enumerate(eps_s):
    traj, info = run_exp_cont(eps=eps, goal_length=N)
    plot_traj_and_qr(traj, info['qr'], eps, N, ax=axs[i,0], n_step=100)
    plot_traj_slices(info['qr'], axs[i,1], eps, n_steps=100)

plt.savefig('fig/exp_cont_conjoined.png')

# <codecell>
### BENCHMARKS
# n_iters = 10
# Ns = [3, 5, 10]
# eps = np.linspace(-2, 2, num=5)

n_iters = 5
Ns = [3, 5, 10]
eps = np.linspace(-2, 2, num=5)
# max_steps = 500

T = 3
lr = 0.1
alpha = 0.1
beta = 1
k = 5

raw_data = []

for N in tqdm(Ns):
    for e in eps:
        cases = [
            Case('Adaptive', run_exp_cont, {'eps': e, 'goal_length': N}, []),
            Case('Incremental', run_inc_cont, {'eps': e, 'goal_length': N}, []),
            # Case('Random', run_random, {'eps': e, 'goal_length': N, 'is_cont': True}, []),
            # Case('Final', run_final_task_only, {'eps': e, 'goal_length': N, 'is_cont': True}, []),
        ]

        run_exp(n_iters=n_iters, cases=cases, max_steps=N * 100, lr=lr, T=T)
        raw_data.extend(cases)

df = pd.DataFrame(raw_data)

# <codecell>
fig_dir = Path('fig/benchmark_cont')
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

# <codecell>

for N in Ns:
    plot_df = df.apply(extract_plot_vals, axis=1).explode('traj_lens')
    plot_df = plot_df.loc[plot_df['N'] == N]

    ax = sns.barplot(plot_df, x='eps', y='traj_lens', hue='name')
    ax.get_legend().set_title(None)
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Steps')
    ax.set_title(f'N = {N}')
    sns.move_legend(ax, 'upper right')

    plt.savefig(fig_dir / f'N_{N}.png')
    plt.clf()

# <codecell>
### BENCHMARK CONJOINED
plot_df = df.apply(extract_plot_vals, axis=1).explode('traj_lens')

g = sns.catplot(plot_df, x='eps', y='traj_lens', hue='name', col='N', kind='bar', sharey=False, height=3, aspect=1.4)
g.set_axis_labels(f'$\epsilon$', 'Steps')
g.legend.set_title('')

plt.savefig(fig_dir / 'cont_benchmarks_conjoined.png')
# %%
