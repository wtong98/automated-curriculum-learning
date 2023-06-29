"""
Estimate Q values from test performance on individual environments

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import functools
import re

import sys
sys.path.append('../')

from meander_bench import *

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

models_dirs = [
    ('adp', Path('remote/trail_sample/adp/0')),
    ('inc', Path('remote/trail_sample/inc/0'))
]

# <codecell>
if __name__ == '__main__':
    n_procs = 24
    n_iters = 5

    for name, model_dir in models_dirs:
        print('Processing', name)

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

        np.save(f'meander_{name}_probs.npy', res)
    
# <codecell>
probs = np.load('remote/trail_sample/meander_inc_probs.npy')

plt.gcf().set_size_inches(8, 2)
ax = plt.gca()

qr = np.copy(probs)
qr = np.flip(qr.T, axis=0)
im = ax.imshow(qr, aspect='auto')

plt.gcf().tight_layout()
# %%
