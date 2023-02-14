"""
Visualizations for continuous methods
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
### BENCHMARK: Incremental vs. Our algorithms
# Incremental
# Adaptive oscillator
# Adaptive exponential

n_iters = 5
Ns = [3, 5, 10, 20]
eps = np.linspace(-4, 2, num=7)
max_steps_fac = 200

# n_iters = 1
# Ns = [3, 5]
# eps = np.linspace(-1, 1, num=3)

T = 3
lr = 0.1

all_cases = defaultdict(lambda: None)
raw_data = []

for N in tqdm(Ns):
    for e in eps:
        cases = [
            Case('Incremental', run_incremental_perfect, {'eps': e, 'goal_length': N}, []),
            Case('Adaptive (Osc)', run_adp_cont, {'eps': e, 'goal_length': N}, []),
            Case('Adaptive (Exp)', run_adp_exp_cont, {'eps': e, 'goal_length': N}, []),
        ]

        run_exp(n_iters=n_iters, cases=cases, max_steps=N * 200, lr=lr, T=T)
        all_cases[(N, e)] = cases
        raw_data.extend(cases)

df = pd.DataFrame(raw_data)

# <codecell>
fig_dir = save_path / 'inc_v_adp_cont'
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


plot_df = df.apply(extract_plot_vals, axis=1).explode('traj_lens')

for N in Ns:
    curr_df = plot_df.loc[plot_df['N'] == N]

    ax = sns.barplot(curr_df, x='eps', y='traj_lens', hue='name')
    ax.get_legend().set_title(None)
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Iterations')
    ax.set_yscale('log')
    ax.set_title(f'N = {N}')
    sns.move_legend(ax, 'upper right')

    plt.savefig(fig_dir / f'N_{N}.svg')
    plt.clf()

# <codecell>
### PLOT Q VALUES
fig_dir = save_path / 'q_vals_cont'
if not fig_dir.exists():
    fig_dir.mkdir(parents=True)

for _, row in tqdm(df.iterrows(), total=len(df)):
    N = row['run_params']['goal_length']
    e = row['run_params']['eps']

    N_cont, e_cont = to_cont(N, e)
    
    for i, (traj, info) in enumerate(zip(row['runs'], row['info'])):
        qr = np.array(info['qr'])
        qr = np.flip(qr.T, axis=0)

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(traj)
        axs[0].set_yticks(np.arange(N) + 1)
        axs[0].set_xlim((0, len(traj)))
        axs[0].set_ylabel('N')

        axs[1].imshow(qr, aspect='auto')
        axs[1].set_yticks(np.arange(N), np.flip(np.arange(N) + 1))
        axs[1].set_ylabel('N')
        axs[1].set_xlabel('Iterations')

        name = row['name']
        fig.suptitle(f'{name}: N = {N}, eps = {e}')
        plt.savefig(fig_dir / f'{name}_N_{N}_eps_{e}_{i}.svg')
        plt.clf()
        plt.close(fig)


# <codecell>
### COMPARE TRAJECTORIES
fig_dir = save_path / 'traj_cont'
if not fig_dir.exists():
    fig_dir.mkdir(parents=True)

n_iters = 3

params_df = pd.DataFrame(df['run_params'].tolist())

for N in Ns:
    for e in eps:
        plot_df = df.loc[(params_df['goal_length'] == N) & (params_df['eps'] == e)]

        for i, (_, row) in enumerate(plot_df.iterrows()):
            traj = row['runs']
            xs = np.arange(len(traj[0])) + 0.15 * i
            plt.plot(xs, traj[0], label=row['name'], alpha=0.7)
            plt.xlabel('Iteration')
            plt.ylabel('N')
            plt.title(f'eps = {e:.2f}')

        plt.legend()
        plt.savefig(fig_dir / f'N_{N}_eps_{e}.svg')
        plt.clf()
