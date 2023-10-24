"""
Estimate Q values from test performance on individual environments

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import functools
import re

import sys
sys.path.append('../')

import matplotlib

from meander_bench import *

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

sched = [
    (10, [(0.5, 0.6)]),
    (30, [(0.5, 0.6)]),
    (50, [(0.5, 0.6)]),
    (70, [(0.5, 0.6)]),
    (90, [(0.5, 0.6)]),
    (100, [(0.5, 0.63)])
]
sched = to_sched(*zip(*sched))

def test_student(student, env, n_iters=5):
    total_success = 0
    
    for _ in range(n_iters):
        obs = env.reset()
        is_done = np.array(env.num_envs * [False])
        is_success = np.zeros(env.num_envs)

        while not np.all(is_done):
            action, _ = student.predict(obs, deterministic=True)
            obs, _, is_done_curr, info = env.step(action)
            is_success_curr = np.array([i['is_success'] for i in info])

            is_success[~is_done] = is_success_curr[~is_done]
            is_done[~is_done] = is_done_curr[~is_done]

        total_success += np.sum(is_success)
    
    return total_success / (n_iters * len(is_done))


def env_fn(args): 
    return TrailEnv(trail_map=MeanderTrail(**args))

save_dir = Path('/n/holyscratch01/pehlevan_lab/Lab/wlt/acl/trail_runs/trained')

models_dirs = [
    ('adp', save_dir / 'adp/all'),
    ('inc', save_dir / 'inc/all')
]

# <codecell>
if __name__ == '__main__':
    n_procs = 24
    n_iters = 5

    for name, teacher_dir in models_dirs:
        print('Processing', name)

        for model_dir in teacher_dir.iterdir():
            print('reading', model_dir.name)

            save_files = list(model_dir.iterdir())
            n_saves = len(save_files)
            n_lvls = len(sched)

            res = np.zeros((n_saves, n_lvls))
            for fp in tqdm(save_files):
                if fp.name == 'gen_final.zip':
                    save_idx = n_saves - 1
                else:
                    num = re.findall(r'\d+', fp.name)[0]
                    save_idx = int(num) - 1
                
                for i, args in enumerate(sched):
                    env = SubprocVecEnv(n_procs * [functools.partial(env_fn, args=args)])
                    model = make_model(env)
                    model.set_parameters(fp)

                    succ = test_student(model, env, n_iters=n_iters)
                    res[save_idx, i] = succ

            np.save(f'meander_{name}_probs.{model_dir.name}.npy', res)
# <codecell>
def plot_estimates(df, probs, name, save_path=None):
    idx = None
    if name == 'adp':
        idx = 0
    elif name == 'inc':
        idx = 1
    else:
        raise ValueError('name should be "adp" or "inc"')

    traj = np.array([t for t in df.runs[idx][0]])

    plt.gcf().set_size_inches(8, 3)
    ax = plt.gca()

    ticks = np.arange(6)
    ax.set_yticks(ticks, np.flip(np.arange(6) + 1))
    ax.set_ylabel('N')
    ax.set_xlabel('Steps')

    ratio = probs[:,1:] / (probs[:,:-1] + 1e-8)
    ratio = np.concatenate((probs[:,[0]], ratio), axis=1)
    ratio = np.clip(ratio, 0.01, 0.99)
    qr = np.log(ratio / (1 - ratio))

    qr = probs
    qr = np.flip(qr.T, axis=0)
    ax.imshow(qr, aspect='auto')

    ax.plot(4.6 - traj, color='red', linewidth=3)

    plt.gcf().tight_layout()

    if save_path is not None:
        plt.savefig(save_path)


estm_dir = Path('remote/trail_sample/estimates')
estimates = {}

for fp in estm_dir.iterdir():
    if fp.suffix != '.npy':
        continue

    name, idx, _ = fp.name.split('.')
    _, teacher_name, _ = name.split('_')
    if idx not in estimates:
        estimates[idx] = {}

    estimates[idx][teacher_name] = np.load(fp)

for fp in estm_dir.iterdir():
    if fp.suffix != '.pkl':
        continue
    
    _, _, idx = fp.stem.split('_')
    estimates[idx]['df'] = pd.read_pickle(fp)

save_dir = Path('remote/trail_sample/plots')
if not save_dir.exists():
    save_dir.mkdir()

for idx, res in estimates.items():
    print(res.keys())
    for name in ('inc', 'adp'):
        plot_estimates(res['df'], res[name], name, save_path=save_dir / f'meander_{idx}_{name}.png')
        plt.clf()
    
# <codecell>
# TODO: rework smoothing scheme for improving visualization
df = pd.read_pickle('remote/trail_sample/meander_results.pkl')
traj = np.array(df.runs[1][0])

probs = np.load('remote/trail_sample/meander_inc_probs.npy')

plt.gcf().set_size_inches(10, 2)
ax = plt.gca()

ticks = np.arange(6)
ax.set_yticks(ticks, np.flip(np.arange(6) + 1))
ax.set_ylabel('N')
ax.set_xlabel('Steps')

# ratio = probs[:,1:] / (probs[:,:-1] + 1e-8)
# ratio = np.concatenate((probs[:,[0]], ratio), axis=1)
# ratio = np.clip(ratio, 0.01, 0.99)
# qr = np.log(ratio / (1 - ratio))


log_probs = np.log(probs + 1e-10)
qr = log_probs[:,1:] - log_probs[:,:-1]
qr = np.concatenate((log_probs[:,[0]], qr), axis=1)
# qr = np.log(probs[:,1:] + 1e-8) - np.log(probs[:,:-1] + 1e-8)

qr = np.flip(qr.T, axis=0)
# im = ax.imshow(qr, aspect='auto', vmin=0, vmax=1.1, cmap='viridis')
im = ax.imshow(qr, aspect='auto', vmin=-1, vmax=1, cmap='viridis')

ax.plot(4.6 - traj, color='C3', linewidth=2)

plt.colorbar(im)
plt.gcf().tight_layout()
# plt.savefig('fig/trail_inc_probs.png')
# %%