"""
Final figures for the paper (Figure 2)
"""

# <codecell>
import sys
sys.path.append('../')

from env import *

# TODO: tune
class TeacherExpIncremental(Agent):
    def __init__(self, tau=0.95, discount=0.8) -> None:
        super().__init__()
        self.tau = tau
        self.discount = discount
        self.avgs = []
    
    def next_action(self, state):
        _, trans = state
        self._consume_trans(trans)

        if self.avgs[-1] > self.tau:
            return 2
        return 1
    
    def reset(self):
        self.avgs = []

    def _consume_trans(self, trans):
        avg = self.avgs[-1] if len(self.avgs) > 0 else 0
        for x in trans:
            avg = (1 - self.discount) * x + self.discount * avg
        self.avgs.append(avg)


def run_exp_inc(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, **teacher_kwargs):
    teacher = TeacherExpIncremental(**teacher_kwargs)
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, return_transcript=True)
    traj = [env.N]
    env.reset()
    all_qr = []

    obs = (1, [])
    for _ in range(max_steps):
        action = teacher.next_action(obs)
        obs, _, is_done, _ = env.step(action)
        traj.append(env.N)

        qr = [env.student.q_r[i] for i in range(goal_length)]
        all_qr.append(qr)

        if is_done:
            break
    
    return traj, {'qr': all_qr, 'teacher': teacher}


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
def plot_traj_and_qr(traj, qr, eps, save_path=None):
    plt.clf()
    plt.gcf().set_size_inches(8, 3)

    qr = np.array(qr)
    qr = np.flip(qr.T, axis=0) + eps
    plt.imshow(qr, aspect='auto', vmin=0, vmax=10)
    plt.yticks(np.arange(N), np.flip(np.arange(N) + 1))
    plt.ylabel('N')
    plt.xlabel('Steps')
    plt.title(fr'$\epsilon = {eps}$')

    plt.colorbar()

    plt.plot(10 - np.array(traj), color='red')
    plt.xlim((0, len(traj) - 1.5))

    if save_path:
        plt.savefig(save_path)


N = 10
eps = -0.5
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
