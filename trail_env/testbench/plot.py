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

ax = sns.barplot(plot_df, x='name', y='traj_lens', color='C0')
ax.set_ylabel('Iterations')
ax.set_xlabel('')
ax.set_title(f'Meandering Trail Benchmark')

plt.gcf().tight_layout()
plt.savefig('fig/meander_bench.png')
# plt.clf()

# <codecell>
### PLOT TRAJECTORIES
for k, row in df.iterrows():
    for i, traj in enumerate(row['runs']):
        if i == 0:
            plt.plot(traj, label=row['name'], color=f'C{k}', alpha=0.5)
        else:
            plt.plot(traj, color=f'C{k}', alpha=0.5)

plt.legend()

# <codecell>
df = pd.read_pickle('remote/plume_results.pkl')

def extract_plot_vals(row):
    traj_lens = [len(traj) for traj in row['runs']]

    return pd.Series([
        row['name'],
        traj_lens
    ], index=['name', 'traj_lens'])


plot_df = df.apply(extract_plot_vals, axis=1).explode('traj_lens')

ax = sns.barplot(plot_df, x='name', y='traj_lens', color='C0')
ax.set_ylabel('Iterations')
ax.set_xlabel('')
ax.set_title(f'Plume Benchmark')

plt.gcf().tight_layout()
# plt.savefig('fig/plume_bench.png')
# plt.clf()

# <codecell>
### PLOT TRAJECTORIES
for k, row in df.iterrows():
    for i, traj in enumerate(row['runs']):
        if i == 0:
            plt.plot(traj, label=row['name'], color=f'C{k}', alpha=0.5)
        else:
            plt.plot(traj, color=f'C{k}', alpha=0.5)

plt.legend()
# %%
