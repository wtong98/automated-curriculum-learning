"""
Plotting benchmark results
"""

# <codecell>
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append('../')

# <codecell>
df = pd.read_pickle('remote/meander_results.pkl')

def extract_plot_vals(row):
    traj_lens = [len(traj) for traj in row['runs']]

    return pd.Series([
        row['name'],
        traj_lens
    ], index=['name', 'traj_lens'])


plot_df = df.apply(extract_plot_vals, axis=1).explode('traj_lens')

fig, axs = plt.subplots(1, 2, figsize=(8, 3))

ax = sns.barplot(plot_df, x='name', y='traj_lens', ax=axs[0])
ax.set_ylabel('Steps')
ax.set_xlabel('')

x_labs = ax.get_xticklabels()
x_labs[0] = 'Adaptive'
ax.set_xticklabels(x_labs)
ax.set_title('Benchmarks')
# ax.set_title(f'Meandering Trail Benchmark')
# plt.clf()

### PLOT TRAJECTORIES
for k, row in df.iterrows():
    if k == 2:
        break

    for i, traj in enumerate(row['runs']):
        if i == 0:
            axs[1].plot(traj, label=row['name'], color=f'C{k}', alpha=0.5)
        else:
            axs[1].plot(traj, color=f'C{k}', alpha=0.6)
    
axs[1].set_xlabel('Steps')
axs[1].set_ylabel('Difficulty index')
axs[1].legend(loc='lower right')
axs[1].set_title('Trajectory')

fig.tight_layout()
plt.savefig('fig/meander_bench.png')

# <codecell>
df = pd.read_pickle('remote/plume_results.pkl')

def extract_plot_vals(row):
    traj_lens = [len(traj) for traj in row['runs']]

    return pd.Series([
        row['name'],
        traj_lens
    ], index=['name', 'traj_lens'])


plot_df = df.apply(extract_plot_vals, axis=1).explode('traj_lens')


fig, axs = plt.subplots(1, 2, figsize=(8, 3))

ax = sns.barplot(plot_df, x='name', y='traj_lens', ax=axs[0])
ax.set_ylabel('Iterations')
ax.set_xlabel('')
ax.set_title(f'Benchmarks')

x_labs = ax.get_xticklabels()
x_labs[0] = 'Adaptive'
ax.set_xticklabels(x_labs)

# plt.clf()

### PLOT TRAJECTORIES

for k, row in df.iterrows():
    if k == 2:
        break

    for i, traj in enumerate(row['runs']):
        if i == 0:
            axs[1].plot(traj, label=row['name'], color=f'C{k}', alpha=0.5)
        else:
            axs[1].plot(traj, color=f'C{k}', alpha=0.6)
    
axs[1].set_xlim(xmax=60)
axs[1].set_xlabel('Steps')
axs[1].set_ylabel('Difficulty index')
axs[1].legend(loc='lower right')
axs[1].set_title('Trajectory')

fig.tight_layout()
plt.savefig('fig/plume_bench.png')
# %%
