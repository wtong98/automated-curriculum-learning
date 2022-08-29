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

from env import TrailEnv
from curriculum import *

def make_model(env):
    return PPO("CnnPolicy", env, verbose=1,
                n_steps=1024,
                batch_size=256,
                ent_coef=0.1,
                gamma=0.98,
                gae_lambda=0.9,
                clip_range=0.2,  # TODO: try using sched?
                max_grad_norm=1,
                vf_coef=0.36,
                n_epochs=5,
                learning_rate=0.0001,
                tensorboard_log='log',
                policy_kwargs={
                    'net_arch': [{'pi': [128, 128], 'vf': [128, 128]}],
                    'activation_fn': torch.nn.ReLU
                },
                device='cpu'
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

def to_sched(len_sched, break_sched):
    trail_args = {
        'width': 5,
        'diff_rate': 0.02,
        'radius': 70,
        'reward_dist': -1,
        'range': (-np.pi, np.pi)
    }

    sched = [dict(length=l, breaks=b, **trail_args) for l, b in zip(len_sched, break_sched)]
    return sched

def to_sched_cont():
    trail_args = {
        'width': 5,
        'diff_rate': 0.02,
        'radius': 70,
        'reward_dist': -1,
        'range': (-np.pi, np.pi)
    }

    def sched(x):
        return dict(length=x, breaks=[(0.5, 0.6)], **trail_args)
    
    return sched


Case = namedtuple('Case', ['name', 'teacher', 'teacher_params', 'cb_params', 'traj'])


if __name__ == '__main__':
    n_runs = 1
    # sched = make_break_sched(8, start_len=80, end_len=160, inc=0.02)
    sched = [
        (10, [(0.5, 0.6)]),
        (20, [(0.5, 0.6)]),
        (30, [(0.5, 0.6)]),
        (40, [(0.5, 0.6)]),
        (50, [(0.5, 0.6)]),
        (60, [(0.5, 0.6)]),
        (60, [(0.5, 0.61)]),
        (70, [(0.5, 0.6)]),
        (70, [(0.5, 0.61)]),
        (80, [(0.5, 0.6)]),
        (80, [(0.5, 0.61)]),
        (90, [(0.5, 0.6)]),
        (90, [(0.5, 0.61)]),
        (100, [(0.5, 0.6)]),
        (110, [(0.5, 0.6)]),
        (120, [(0.5, 0.6)]),
        (120, [(0.5, 0.61)]),
        (120, [(0.5, 0.62)]),
        (120, [(0.5, 0.63)]),
        (120, [(0.5, 0.64)]),
        (120, [(0.5, 0.65)]),
        (120, [(0.5, 0.66)]),
        (120, [(0.5, 0.67)]),
        (120, [(0.5, 0.68)]),
        (120, [(0.5, 0.69)]),
        (120, [(0.5, 0.7)]),
    ]
    # sched = to_sched(*zip(*sched))
    sched = to_sched_cont()

    def env_fn(): return TrailEnv(None)
    env = SubprocVecEnv([env_fn for _ in range(8)])
    eval_env = env_fn()
    
    # TODO: should've jumped by now?
    cases = [
        # Case('Incremental', IncrementalTeacher, {'len_sched': len_sched}, []),
        # Case('Oscillator', OscillatingTeacher, {'sched': sched, 'tau': 0.9, 'conf':0.5}, {'save_every': 1, 'save_path': 'trained/osc_break_ii'}, []),
        # Case('Naive', NaiveTeacher, {'len_sched': len_sched}, [])
        Case('Adaptive', AdaptiveTeacher, {'goal_length': 120, 'sched': sched, 'tau': 0.3}, {'save_every': 1, 'save_path': 'trained/adp'}, []), # TODO: monitor probabilities (some seem impossible?)
    ]

    for i in tqdm(range(n_runs)):
        for case in cases:
            teacher = case.teacher(**case.teacher_params)
            model = make_model(env)
            # model = PPO.load('trained/osc_break/0/gen93')
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
    plt.savefig('trained/osc_break_ii/0/tt_trajs.png')


# %%  SHOWCASE PERFORMANCE IN PLOTS
save_path = Path('trained/adp/0/')
max_gen = 68

# trail_args = {
#     'length': 80,
#     'width': 5,
#     'diff_rate': 0.01,
#     'radius': 100,
#     'reward_dist': -1,
#     'range': (-np.pi, np.pi)
# }
trail_args = sched(120)

for i in tqdm(range(1, max_gen + 1)):
    model_path = save_path / f'gen{i}'
    # print('loading model')
    model = PPO.load(model_path, device='cpu')

    n_runs = 8
    headings = np.linspace(-np.pi, np.pi, num=n_runs)

    maps = []
    position_hists = []

    # print('preparing to generate headings')
    for heading in headings:
        trail_map = MeanderTrail(**trail_args, heading=heading)
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

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    for ax, m, position_history in zip(axs.ravel(), maps, position_hists):
        m.plot(ax=ax)
        ax.plot(*zip(*position_history), linewidth=2, color='black')

    fig.suptitle('Sample of agent runs')
    fig.tight_layout()
    plt.savefig(save_path / f'gen{i}.png')
    plt.clf()


# <codecell> SINGLE PROBE
model_path = Path('trained/osc_break_ii/0/gen172.zip')

# trail_args = {
#     'length': 160,
#     'width': 5,
#     'diff_rate': 0.01,
#     'radius': 100,
#     'reward_dist': -1,
#     'range': (-np.pi, np.pi),
#     'breaks':[(0.5, 0.53)]
# }
trail_args = sched(120)

model = PPO.load(model_path, device='cpu')

n_runs = 8
headings = np.linspace(-np.pi, np.pi, num=n_runs)

maps = []
position_hists = []

# print('preparing to generate headings')
for heading in headings:
    trail_map = MeanderTrail(**trail_args, heading=heading)
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

fig, axs = plt.subplots(2, 4, figsize=(16, 8))

for ax, m, position_history in zip(axs.ravel(), maps, position_hists):
    m.plot(ax=ax)
    ax.plot(*zip(*position_history), linewidth=2, color='black')

fig.suptitle('Sample of agent runs')
fig.tight_layout()
plt.savefig('tmp.png')
# %%
# TMP SINGLE PLOT

model_path = Path('trained/osc_break_ii/0/gen172.zip')

trail_args = sched(120)
model = PPO.load(model_path, device='cpu')

maps = []
position_hists = []

# print('preparing to generate headings')
trail_map = MeanderTrail(**trail_args, heading=0)
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

fig, axs = plt.subplots(1, 1, figsize=(8, 8))

for ax, m, position_history in zip([axs], maps, position_hists):
    m.plot(ax=ax, ymin=-70, ymax=130)
    ax.plot(*zip(*position_history), linewidth=2, color='black')

fig.suptitle('Sample run')
fig.tight_layout()
plt.savefig('sample_run.svg')
# plt.savefig('tmp.png')
