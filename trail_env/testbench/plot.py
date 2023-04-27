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
from env import *
from curriculum import *

class EstimateQValCallback:
    def __init__(self, sched: list, trail_class=MeanderTrail, n_tests=10) -> None:
        self.sched = sched
        self.trail_class = trail_class
        self.probs = []
        self.n_tests = n_tests

    def __call__(self, cb: CurriculumCallback):
        prob_succ = [1]
        prob = 1
        for args in tqdm(self.sched):
            if prob != 0:
                prob = self._test_student(cb.teacher.student, args)
            else:
                print('warn: zero prob, skipping')

            prob_succ.append(prob)

        # prob_succ = np.array([1] + [self._test_student(cb.teacher.student, args) for args in tqdm(self.sched)])
        prob_succ = np.array(prob_succ)
        print('PROBS', prob_succ)

        ratios = prob_succ[1:] / prob_succ[:-1]
        print('RATIOS', ratios)

        qs = logit(ratios)
        print('QS', qs)

        self.probs.append(ratios)

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
### PLOT EST Q-Values (meander)
df = pd.read_pickle('meander_results.pkl')

fig = plt.gcf()
ax = plt.gca()

adp_ratios = np.load('meander_adp_probs.npy')

traj = np.array([0] + df.loc[0]['runs'][0])
data = adp_ratios.T
data[np.isnan(data)] = 0

im = ax.imshow(data, aspect='auto', origin='lower', vmin=0, vmax=1)

ax.plot(traj + 0.45, color='red')
    
ax.set_xlabel('Steps')
ax.set_ylabel('Difficulty index')
ax.legend(loc='lower right')
ax.set_title('Adaptive: Estimated Ratios')

fig.colorbar(im)
fig.tight_layout()
plt.savefig('fig/meander_adp_ratios.png')

# <codecell>
plt.clf()
fig = plt.gcf()
ax = plt.gca()

adp_ratios = np.load('meander_inc_probs.npy')

traj = np.array([0] + df.loc[1]['runs'][0])
data = adp_ratios.T
data[np.isnan(data)] = 0

im = ax.imshow(data, aspect='auto', origin='lower', vmin=0, vmax=1)

ax.plot(traj + 0.45, color='red')
    
ax.set_xlabel('Steps')
ax.set_ylabel('Difficulty index')
ax.legend(loc='lower right')
ax.set_title('Incremental: Estimated Ratios')

fig.colorbar(im)
fig.tight_layout()
plt.savefig('fig/meander_inc_ratios.png')