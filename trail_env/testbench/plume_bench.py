"""
Benchmark performance of various teacher strategies

author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv 
import torch
from tqdm import tqdm

import sys
sys.path.append('../')

from env import TrailEnv
from curriculum import *
from trail_map import *

def make_model(env):
    return PPO("CnnPolicy", env, verbose=1,
                n_steps=1024,
                batch_size=256,
                ent_coef=0.25,
                gamma=0.98,
                gae_lambda=0.9,
                clip_range=0.2,
                max_grad_norm=1,
                vf_coef=0.36,
                n_epochs=5,
                learning_rate=0.0001,
                tensorboard_log='log',
                policy_kwargs={
                    'net_arch': [{'pi': [128, 128], 'vf': [128, 128]}],
                    'activation_fn': torch.nn.ReLU
                },
                device='auto'
                )

def run_session(student, teacher, eval_env, cb_params):
    student.learn(total_timesteps=3000000, 
                  eval_env=eval_env, 
                  eval_freq=512, 
                  callback=[CurriculumCallback(teacher, eval_env=eval_env, **cb_params)])
    return teacher.trajectory

def make_break_sched(n=8, start_len=80, end_len=160, inc=0.025):
    len_sched = [start_len, end_len] + n * [end_len]
    break_sched = [[], []] + [[(0.5, 0.5 + i * inc)] for i in range(1, n + 1)]
    return to_sched(len_sched, break_sched)

def to_sched(rates):
    trail_args = {
        'wind_speed': 5,
        'length_scale': 20,
    }

    sched = [dict(start_rate=r, max_steps='auto', **trail_args) for r in rates]
    return sched

# def to_sched_cont():
#     trail_args = {
#         'width': 5,
#         'diff_rate': 0.02,
#         'radius': 70,
#         'reward_dist': -1,
#         'range': (-np.pi, np.pi)
#     }

#     def sched(x):
#         return dict(length=x, breaks=[(0.5, 0.6)], **trail_args)
    
#     return sched


Case = namedtuple('Case', ['name', 'teacher', 'teacher_params', 'cb_params', 'traj'])


if __name__ == '__main__':
    n_runs = 1
    rates = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.275, 0.25, 0.225, 0.2, 0.175, 0.15, 0.125, 0.1]
    sched = to_sched(rates)

    def env_fn(): return TrailEnv()

    env = SubprocVecEnv([env_fn for _ in range(8)])
    eval_env = env_fn()
    
    cases = [
        Case('Incremental', IncrementalTeacher, {'goal_length': len(rates) - 1, 'sched': sched, 'trail_class': PlumeTrail}, {'save_every': 1, 'save_path': 'trained/plume_rate'}, []),
        # Case('Oscillator', OscillatingTeacher, {'sched': sched, 'tau': 0.9, 'conf':0.5, 'trail_class': PlumeTrail}, {'save_every': 1, 'save_path': 'trained/inc_plume'}, []),
        # Case('Naive', NaiveTeacher, {'len_sched': len_sched}, [])
        # Case('Adaptive', AdaptiveTeacher, {'goal_length': 120, 'sched': sched, 'tau': 0.3}, {'save_every': 1, 'save_path': 'trained/adp'}, []), 
    ]

    for i in tqdm(range(n_runs)):
        for case in cases:
            teacher = case.teacher(**case.teacher_params)
            model = make_model(env)
            model.set_env(env)
            case.cb_params['save_path'] += f'/{i}'
            traj = run_session(model, teacher, eval_env, case.cb_params)
            traj = [t[0] for t in traj]
            case.traj.append(traj)

# <codecell>
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    for i, case in enumerate(cases):
        label = {'label': case.name}
        for run in case.traj:
            axs[0].plot(run, color=f'C{i}', alpha=0.7, **label)
            label = {}

    axs[0].legend()
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Schedule index')
    # axs[0].set_yticks(np.arange(len(sched)))

    # axs[0].set_xlim((800, 900))

    all_lens = [[len(run) for run in case.traj] for case in cases]
    all_means = [np.mean(lens) for lens in all_lens]
    all_serr = [2 * np.std(lens) / np.sqrt(n_runs) for lens in all_lens]
    all_names = [case.name for case in cases]

    axs[1].bar(np.arange(len(cases)), all_means, tick_label=all_names, yerr=all_serr)
    axs[1].set_ylabel('Iterations')

    fig.suptitle(f'Trail sched')
    fig.tight_layout()
    plt.savefig('trained/inc_plume/0/tt_trajs.png')

# '''

# %%  SHOWCASE PERFORMANCE IN PLOTS
# TODO: mark odor along trajectory of agent
save_path = Path('trained/plume_rate/0/')
max_gen = 98

# trail_args = {
#     'length': 80,
#     'width': 5,
#     'diff_rate': 0.01,
#     'radius': 100,
#     'reward_dist': -1,
#     'range': (-np.pi, np.pi)
# }
trail_args = sched[-3]

for i in tqdm(list(range(1, max_gen + 1)) + ['_final']):
    model_path = save_path / f'gen{i}'
    # print('loading model')
    model = PPO.load(model_path, device='cpu')

    maps = []
    position_hists = []
    odor_hists = []

    # print('preparing to generate headings')
    for _ in range(8):
        trail_map = PlumeTrail(**trail_args)
        env = TrailEnv(trail_map, discrete=True, treadmill=True)

        obs = env.reset()
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, is_done, _ = env.step(action)

            if is_done:
                break
        
        # print('gen heading')
        maps.append(trail_map)
        position_hists.append(env.agent.position_history)
        odor_hists.append(env.agent.odor_history)

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    for ax, m, position_history, odor_history in zip(axs.ravel(), maps, position_hists, odor_hists):
        hist = np.array(position_history)
        odor_hist = np.array(odor_history)
        odor_hist = odor_hist[odor_hist[:,0] > 0]
        x_min = min(-30, np.min(hist[:,0]))
        x_max = max(30, np.max(hist[:,0]))

        y_min = min(-50, np.min(hist[:,1]))
        y_max = max(10, np.max(hist[:,1]))

        m.plot(ax=ax, x_lim=(x_min, x_max), y_lim=(y_min, y_max))
        ax.plot(*zip(*position_history), linewidth=2, color='black')
        mpb = ax.scatter(odor_hist[:,1], odor_hist[:,2], c=odor_hist[:,0], cmap='summer', s=30, vmin=0, vmax=1)

    fig.suptitle('Sample of agent runs')
    fig.tight_layout()
    plt.savefig(save_path / f'gen{i}.png')
    plt.clf()


# <codecell> SINGLE PROBE
model_path = Path('trained/plume_rate/0/gen30.zip')

trail_args = sched[-6]
# trail_args['start_y'] = -40
model = PPO.load(model_path, device='cpu')

maps = []
position_hists = []
odor_hists = []

# print('preparing to generate headings')
for _ in range(1):
    trail_map = PlumeTrail(**trail_args)
    env = TrailEnv(trail_map, discrete=True, treadmill=True)

    obs = env.reset()
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, is_done, _ = env.step(action)

        if is_done:
            break
    
    # print('gen heading')
    maps.append(trail_map)
    position_hists.append(env.agent.position_history)
    odor_hists.append(env.agent.odor_history)

# fig, axs = plt.subplots(2, 4, figsize=(16, 8))
# fig, axs = plt.subplots(1, 2, figsize=(16, 8))
fig, axs = plt.subplots(1, 1)
axs = np.array([axs])

for ax, m, position_history, odor_history in zip(axs.ravel(), maps, position_hists, odor_hists):
    hist = np.array(position_history)
    odor_hist = np.array(odor_history)
    odor_hist = odor_hist[odor_hist[:,0] > 0]

    x_min = min(-30, np.min(hist[:,0]))
    x_max = max(30, np.max(hist[:,0]))

    y_min = min(-50, np.min(hist[:,1]))
    y_max = max(10, np.max(hist[:,1]))

    m.plot(ax=ax, x_lim=(x_min, x_max), y_lim=(y_min, y_max))
    ax.plot(*zip(*position_history), linewidth=2, color='black')
    # mpb = ax.scatter(odor_hist[:,1], odor_hist[:,2], c=odor_hist[:,0], cmap='summer', s=30, vmin=0, vmax=1)
    # fig.colorbar(mpb, ax=ax)

width = 6
ratio = (y_max - y_min) / (x_max - x_min)

fig.set_size_inches(width, ratio*width)
# fig.suptitle('Sample of agent runs')
fig.tight_layout()
plt.savefig('sample_plume.svg')
# plt.savefig('start_y_40.png')

# <codecell>
'''