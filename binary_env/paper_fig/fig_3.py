"""
POMCP and Adaptive Exp plots
"""

# <codecell>
from pathlib import Path
import pickle

import pandas as pd
import seaborn as sns

import sys
sys.path.append('../')

from common import *
from env import *

# <codecell>
n_iters = 1
results = []
for _ in range(n_iters):
    res = run_pomcp_with_retry(max_retries=5, eps=-2, goal_length=10, gamma=0.95)
    results.append(res)

with open('pomcp_res.pkl', 'wb') as fp:
    pickle.dump(results, fp)

# <codecell>
with open('pomcp_res.pkl', 'rb') as fp:
    results = pickle.load(fp)

traj, info = results[0]
plot_traj_and_qr(traj, info['qr'], -2, 10, save_path='fig/fig3_pomcp_eps=-2.png')
# <codecell>
### Adaptive Exp
N = 10
eps = [-2, 0, 2]

for e in eps:
    traj, info = run_adp_exp_disc(eps=e, goal_length=N)
    plot_traj_and_qr(traj, info['qr'], e, save_path=f'fig/fig3_adp_exp_eps={e}.png')


# <codecell>
### Adaptive Exp conjoined
fig, axs = plt.subplots(3, 1, figsize=(5, 8))

N = 10
eps = [2, 0, -2]

for e, ax in zip(eps, axs):
    traj, info = run_adp_exp_disc(eps=e, goal_length=N)
    plot_traj_and_qr(traj, info['qr'], e, N, ax=ax)

plt.savefig('fig/adp_conjoined.png')

# <codecell>
### BENCHMARK: Us vs. Matiisen
# TODO: tune Matiisen params

n_iters = 10
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
            Case('Adaptive', run_adp_exp_disc, {'eps': e, 'goal_length': N}, []),
            Case('Incremental', run_exp_inc, {'eps': e, 'goal_length': N}, []),
            Case('Online', run_online, {'eps': e, 'goal_length': N, 'alpha': alpha, 'beta': beta}, []),
            Case('Naive', run_naive, {'eps': e, 'goal_length': N, 'alpha': alpha, 'beta': beta, 'k': k}, []),
            Case('Window', run_window, {'eps': e, 'goal_length': N, 'alpha': alpha, 'beta': beta, 'k': k}, []),
            Case('Sampling', run_sampling, {'eps': e, 'goal_length': N, 'alpha': alpha, 'k': k}, []),
            Case('Random', run_random, {'eps': e, 'goal_length': N}, []),
            Case('Final', run_final_task_only, {'eps': e, 'goal_length': N}, []),
        ]

        run_exp(n_iters=n_iters, cases=cases, max_steps=N * 100, lr=lr, T=T)
        raw_data.extend(cases)

df = pd.DataFrame(raw_data)

# <codecell>
df_pomcp = pd.read_pickle('pomcp.pkl')
df_pomcp


df = pd.concat((df_pomcp, df), ignore_index=True)
# <codecell>
fig_dir = Path('fig/us_v_matiisen_v_pomcp')
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

    plt.savefig(fig_dir / f'N_{N}.svg')
    plt.clf()

# <codecell>
### BENCHMARK CONJOINED
plot_df = df.apply(extract_plot_vals, axis=1).explode('traj_lens')

g = sns.catplot(plot_df, x='eps', y='traj_lens', hue='name', col='N', kind='bar', sharey=False, height=3, aspect=1.4)
g.set_axis_labels(f'$\epsilon$', 'Steps')
g.legend.set_title('')

plt.savefig('fig/us_v_matiisen_v_pomcp/seq_benchmarks_conjoined.png')


# %%
### POMCP plots
df = pd.read_pickle('pomcp.pkl')

# <codecell>
for _, run in tqdm(df.iterrows(), total=len(df)):
    params = run['run_params']
    eps = params['eps']
    N = params['goal_length']

    if N == 10:
        for i, (traj, info) in enumerate(zip(run.runs, run['info'])):
            plot_traj_and_qr(traj, info['qr'], eps, N, save_path=f'fig/pomcp/pomcp_N={N}_eps={eps}_id={i}.png')

# <codecell>
fig, axs = plt.subplots(3, 1, figsize=(5, 8))

N = 10
eps = -2
row = df.loc[10]
traj, info = row['runs'][0], row['info'][0]
plot_traj_and_qr(traj, info['qr'], eps, N, ax=axs[2])

N = 10
eps = 0
row = df.loc[12]
traj, info = row['runs'][0], row['info'][0]
plot_traj_and_qr(traj, info['qr'], eps, N, ax=axs[1])

N = 10
eps = 2
row = df.loc[14]
traj, info = row['runs'][0], row['info'][0]
plot_traj_and_qr(traj, info['qr'], eps, N, ax=axs[0])

plt.savefig('fig/pomcp_conjoined.png')