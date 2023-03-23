"""
Final figures for the paper (Figure 2)
"""

# <codecell>
import sys
sys.path.append('../')

from common import *
from env import *

eps = np.linspace(-5, 4, num=25)
N = 10

all_lens = []

n_iters = 5

for _ in range(n_iters):
    trajs = [run_exp_inc(eps=e, goal_length=N) for e in eps]
    traj_lens = [len(t) for t, _ in trajs]
    all_lens.append(traj_lens)

all_lens = np.array(all_lens)

# <codecell>
### Incremental failure plot
mean = np.mean(all_lens, axis=0)
sd_err = np.std(all_lens, axis=0) / np.sqrt(n_iters)

plt.errorbar(eps, traj_lens, fmt='o--', yerr=sd_err, label=r'$\pm 2$ SE')
plt.legend()

plt.xlabel(r'$\epsilon$')
plt.ylabel('Steps')

plt.savefig('fig/fig2_inc_failure.png')

# <codecell>
### Q-value plots
N = 10
eps = 0
traj, info = run_exp_inc(eps=eps, goal_length=N)
plot_traj_and_qr(traj, info['qr'], eps, save_path='fig/fig2_middling_qr.png')

# <codecell>
N = 10
eps = -2
traj, info = run_exp_inc(eps=eps, goal_length=N)
plot_traj_and_qr(traj, info['qr'], eps, save_path='fig/fig2_failure_qs.png')

# <codecell>
N = 10
eps = 2
traj, info = run_exp_inc(eps=eps, goal_length=N)
plot_traj_and_qr(traj, info['qr'], eps, save_path='fig/fig2_success_qr.png')

# %%
