"""
Final figures for the paper (Figure 2)
"""

# <codecell>
import matplotlib as mpl

import sys
sys.path.append('../')

from common import *
from env import *

# <codecell>

eps = np.linspace(-4, 2.5, num=25)
N = 10

all_lens = []

n_iters = 100

for _ in tqdm(range(n_iters)):
    trajs = [run_exp_inc(eps=e, goal_length=N) for e in eps]
    traj_lens = [len(t) for t, _ in trajs]
    all_lens.append(traj_lens)

all_lens = np.array(all_lens)

# <codecell>
### Incremental failure plot
mean = np.mean(all_lens, axis=0)
success_rate = np.mean(all_lens != np.max(all_lens), axis=0)

# sd_err = np.std(all_lens, axis=0) / np.sqrt(n_iters)

# plt.errorbar(eps, traj_lens, fmt='o--', yerr=sd_err, label=r'$\pm 2$ SE')
ax = plt.gca()

ax.plot(eps, mean, '--o')
ax.set_ylabel('Steps', color='C0')
ax.tick_params(axis='y', labelcolor='C0')
ax.set_xlabel(r'$\varepsilon$')

ax2 = ax.twinx()
ax2.plot(eps, success_rate, '--o', color='C1')
ax2.set_ylabel('Success rate', color='C1')
ax2.tick_params(axis='y', labelcolor='C1')


# plt.legend()
plt.savefig('fig/fig2_inc_failure.png')

# <codecell>
### Q-value plots
N = 10
eps = 0
traj, info = run_exp_inc(eps=eps, goal_length=N)
plot_traj_and_qr(traj, info['qr'], eps, N, save_path='fig/fig2_middling_qr.png')

# <codecell>
N = 10
eps = -2
traj, info = run_exp_inc(eps=eps, goal_length=N)
plot_traj_and_qr(traj, info['qr'], eps, N, save_path='fig/fig2_failure_qr.png')

# <codecell>
N = 10
eps = 2
traj, info = run_exp_inc(eps=eps, goal_length=N)
plot_traj_and_qr(traj, info['qr'], eps, N, save_path='fig/fig2_success_qr.png')

# <codecell>
### SAME AS ABOVE, BUT CONJOINED
fig, axs = plt.subplots(3, 2, figsize=(9, 8), sharey=False)

N = 10
eps = 2
traj, info = run_exp_inc(eps=eps, goal_length=N)
plot_traj_and_qr(traj, info['qr'], eps, N, ax=axs[0][0])
plot_traj_slices(info['qr'], ax=axs[0][1], eps=eps)

N = 10
eps = 0
traj, info = run_exp_inc(eps=eps, goal_length=N)
plot_traj_and_qr(traj, info['qr'], eps, N, ax=axs[1][0])
plot_traj_slices(info['qr'], ax=axs[1,1], eps=eps)

N = 10
eps = -2
traj, info = run_exp_inc(eps=eps, goal_length=N)
plot_traj_and_qr(traj, info['qr'], eps, N, ax=axs[2,0])
plot_traj_slices(info['qr'], ax=axs[2,1], eps=eps)

fig.tight_layout()

plt.savefig('fig/inc_fail_conjoined.png')

# <codecell>
### REINFORCEMENT AND FORGETTING WAVES

env = BinaryEnv(5, reward=10)
student = Student(lr=0.1)

incs = 30
iters = 10

cmap = mpl.colormaps['plasma']

for i in range(iters):
    student.learn(env, max_iters=999, max_rounds=incs)
    qs = np.array([student.q_r[i] for i in range(5)])
    plt.plot(1 + np.arange(5), qs + 0.01 * i, 'o--', color=cmap(i/iters), alpha=0.6)

plt.xticks(1 + np.arange(5))

plt.ylabel('Q-value')
plt.xlabel('Level')
plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, incs * iters), cmap=cmap))
plt.gcf().tight_layout()
plt.savefig('reinforcement_wave.png')
# %%
student = Student(lr=0.1)
env = BinaryEnv(5, reward=10)
student.learn(env, max_rounds=500)

# plt.plot([student.q_r[i] for i in range(5)])

env = BinaryEnv(6, reward=10)

incs = 15
iters = 10

for i in range(iters):
    qs = np.array([student.q_r[i] for i in range(6)])
    plt.plot(1 + np.arange(6), qs + 0.01 * i, 'o--', color=cmap(i/iters), alpha=0.6)
    student.learn(env, max_rounds=incs)

plt.xticks(1 + np.arange(6))

plt.ylabel('Q-value')
plt.xlabel('Level')
plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, incs * iters), cmap=cmap))
plt.gcf().tight_layout()
plt.savefig('extinction_wave.png')