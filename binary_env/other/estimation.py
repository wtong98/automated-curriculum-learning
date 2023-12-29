"""
Project files for BDA

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import *
from env import *

np.random.seed(0)

def plot_prediction_vs_true(traj, info, true_eps, ax, x_lim=1000):
    t = info['teacher']

    true_sn = []
    for n, qs in zip(traj, np.array(info['qr'])):
        sn = np.prod(sig(qs[:n] + true_eps))
        true_sn.append(sn)

    if hasattr(t, 'bounds'):
        lwr_95, lwr_50, upr_50, upr_95 = zip(*t.bounds)
        ax.fill_between(np.arange(len(lwr_95))[:x_lim], lwr_95[:x_lim], upr_95[:x_lim], color='C0', alpha=0.15, label='95% Int')
        ax.fill_between(np.arange(len(lwr_50))[:x_lim], lwr_50[:x_lim], upr_50[:x_lim], color='C0', alpha=0.35, label='50% Int')

    ax.plot(t.avgs[:x_lim], color='C0', label='Estimated', linewidth=1.5)
    ax.plot(true_sn[:-1][:x_lim], color='red', label='True', linewidth=1.5)
    ax.set_xlabel('Step', fontsize=18)
    ax.set_ylabel('$\hat{s}_n$', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=14)

N = 10
eps = [-2, 0, 2]
g = 0.8

fig, axs = plt.subplots(1, 3, figsize=(12, 2.5))

for e, ax in zip(eps, axs):
    traj, info = run_exp_inc(eps=e, goal_length=N, discount=g)
    plot_prediction_vs_true(traj, info, e, ax, x_lim=200)
    ax.set_title(fr'$\varepsilon={e}$', fontsize=20)
    
fig.tight_layout()
plt.savefig('fig/ema_estimate.svg')
plt.show()


# <codecell>

np.random.seed(0)

N = 10
eps = [-2, 0, 2]

fig, axs = plt.subplots(1, 3, figsize=(12, 2.5))

for e, ax in zip(eps, axs):
    traj, info = run_beta_inc(eps=e, goal_length=N)
    plot_prediction_vs_true(traj, info, e, ax, x_lim=200)
    ax.set_title(fr'$\varepsilon={e}$', fontsize=20)
    
fig.tight_layout()
plt.savefig('fig/beta_estimate.svg')
plt.show()

# <codecell>
np.random.seed(2)

N = 10
eps = [-2, 0, 2]

fig, axs = plt.subplots(1, 3, figsize=(12, 2.5))
all_info = []

for e, ax in zip(eps, axs):
    traj, info = run_particle_inc(eps=e, goal_length=N)
    plot_prediction_vs_true(traj, info, e, ax, x_lim=200) # TODO: rerun and paste
    ax.set_title(fr'$\varepsilon={e}$', fontsize=20)
    all_info.append(info)
    
fig.tight_layout()
plt.savefig('fig/particle_estimate.svg')

# <codecell>
def plot_particle_diagnostics(info, true_eps, ax, x_lim=200):
    est_qs = []
    est_eps = []

    true_qs = np.array(info['qr'])[:x_lim]

    for p_set in info['teacher'].all_particles:
        qs, eps = zip(*p_set)
        est_qs.append(qs)
        est_eps.append(eps)
        
    est_qs = np.array(est_qs)[:x_lim]
    est_eps = np.array(est_eps)[:x_lim]

    est_qs_bounds = np.quantile(est_qs, (0.025, 0.975), axis=1)
    est_eps_bounds = np.quantile(est_eps, (0.025, 0.975), axis=1)


    xs = np.arange(est_qs_bounds.shape[1])[:x_lim]
    for i in range(est_qs_bounds.shape[-1]):
        ax.fill_between(xs, est_qs_bounds[0,:,i] ,est_qs_bounds[1,:,i], color=f'C{i}', alpha=0.2)
        ax.plot(xs[:-1]+1, true_qs[:-1,i], color=f'C{i}', label=f'$q_{i+1}$')

    ax.fill_between(xs, est_eps_bounds[0,:] ,est_eps_bounds[1,:], color=f'red', alpha=0.2)
    ax.axhline(y=true_eps, color=f'red', label=fr'$\varepsilon$')
    ax.set_ylim(-5, 10.5)

    ax.set_xlabel('Step', fontsize=20)
    ax.set_ylabel(r'$q$', fontsize=20)
    ax.tick_params(labelsize=18)
    ax.set_title(fr'$\varepsilon={true_eps}$', fontsize=22)

    ax.legend()

fig, axs = plt.subplots(1, 3, figsize=(17, 3.5))

for info, e, ax in zip(all_info, eps, axs):
    plot_particle_diagnostics(info, e, ax)

fig.tight_layout()
plt.savefig('fig/particle_diagnostics.svg')

# <codecell>

### BENCHMARKS
n_iters = 10
Ns = [10]
eps = np.linspace(-1, 1, num=3)

T = 3
lr = 0.1

raw_data = []

for N in Ns:
    print('RUNNING', N)
    for e in tqdm(eps):
        cases = [
            Case('EMA', run_exp_inc, {'eps': e, 'goal_length': N}, []),
            Case('Beta', run_beta_inc, {'eps': e, 'goal_length': N}, []),
            Case('Particle', run_particle_inc_with_retry, {'eps': e, 'goal_length': N}, []),
            Case('Random', run_random, {'eps': e, 'goal_length': N}, []),
            # Case('Final', run_final_task_only, {'eps': e, 'goal_length': N}, []),
        ]

        run_exp(n_iters=n_iters, cases=cases, max_steps=N * 100, lr=lr, T=T)
        raw_data.extend(cases)

df = pd.DataFrame(raw_data)

# <codecell>
# from cached POMCP database
df_pomcp = pd.read_pickle('pomcp.pkl')
df = pd.concat((df_pomcp, df), ignore_index=True)
# <codecell>

def extract_plot_vals(row):
    traj_lens = [len(traj) for traj in row['runs']]

    return pd.Series([
        row['name'],
        row['run_params']['goal_length'],
        np.round(row['run_params']['eps'], decimals=2),
        traj_lens
    ], index=['name', 'N', 'eps', 'traj_lens'])

plot_df = df.apply(extract_plot_vals, axis=1).explode('traj_lens')
plot_df = plot_df[np.abs(plot_df['eps']) != 2]

plt.gcf().set_size_inches(4.5, 2.5)
g = sns.barplot(plot_df, x='eps', y='traj_lens', hue='name', errorbar=None, width=0.5, gap=0.2)

g.set_xlabel(r'$\varepsilon$', fontsize=20)
g.set_ylabel('Steps', fontsize=20)
g.tick_params(labelsize=16)
# g.set_yticks([0, 500, 1000])
g.set_xticklabels([-1, 0, 1])
g.legend().set_title('')

g.spines['top'].set_visible(False)
g.spines['right'].set_visible(False)
g.spines['bottom'].set_linewidth(1.25)
g.spines['left'].set_linewidth(1.25)

plt.gcf().tight_layout()
plt.savefig(f'fig/comparison.svg')
plt.show()

# plt.savefig('fig/comparison.png')

# row = df.loc[6]
# plot_pomcp_diagnostics(row['info'][2], row['run_params']['eps'])
# plt.xlabel('Steps')
# plt.ylabel('Q-values')

# plt.tight_layout()
# plt.savefig('fig/pomcp_diagnostic.png')
# %%
