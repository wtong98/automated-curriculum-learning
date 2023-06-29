"""
Benchmark performance of various teacher strategies

author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>
import os
from dataclasses import dataclass, field
from pathlib import Path

from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv 
import torch
from tqdm import tqdm

import sys
sys.path.append('../')

from env import TrailEnv
from curriculum import *

def make_model(env):
    return PPO("CnnPolicy", env, verbose=1,
                n_steps=1024,
                batch_size=256,
                ent_coef=0.1,
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

def run_session(student, teacher, eval_env, cb_params, max_steps=3000000):
    student.learn(total_timesteps=max_steps, 
                #   eval_env=eval_env, 
                #   eval_freq=512, 
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

def logit(x):
    return np.log(x / (1 - x))

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


    def _test_student(self, student, trail_args):
        total_success = 0
        env = TrailEnv(self.trail_class(**trail_args))

        for _ in range(self.n_tests):
            is_done = False
            obs = env.reset()
            while not is_done:
                action, _ = student.predict(obs, deterministic=True)
                obs, _, is_done, info = env.step(action)
                is_success = info['is_success']
            
            if is_success:
                total_success += 1
        
        return total_success / self.n_tests


@dataclass
class Case:
    name: str = ''
    teacher: Callable = None
    teacher_params: dict = field(default_factory=dict)
    cb_params: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)

if __name__ == '__main__':
    save_dir = Path('trail_runs')
    
    scratch_dir = os.getenv('SCRATCH')
    if scratch_dir:
        save_dir = Path(scratch_dir) / 'pehlevan_lab' / 'Lab' / 'wlt' / 'acl' / save_dir

    if not save_dir.exists():
        save_dir.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(None)
    run_id = rng.integers(999_999_999)
    print('RUN ID', run_id)

    n_runs = 1
    # sched = make_break_sched(8, start_len=80, end_len=160, inc=0.02)
    sched = [
        # (5, []),
        # (10, []),
        # (15, []),
        
        (10, [(0.5, 0.6)]),
        (30, [(0.5, 0.6)]),
        (50, [(0.5, 0.6)]),
        (70, [(0.5, 0.6)]),
        (90, [(0.5, 0.6)]),
        (100, [(0.5, 0.63)]),
        # (100, [(0.5, 0.66)]),

        # (110, [(0.5, 0.6)]),
        # (85, [(0.5, 0.6)]),

        # (10, [(0.5, 0.6)]),
        # (20, [(0.5, 0.6)]),
        # (30, [(0.5, 0.6)]),
        # (40, [(0.5, 0.6)]),
        # (50, [(0.5, 0.6)]),
        # (60, [(0.5, 0.6)]),
        # (60, [(0.5, 0.61)]),
        # (70, [(0.5, 0.6)]),
        # (70, [(0.5, 0.61)]),
        # (80, [(0.5, 0.6)]),


        # (80, [(0.5, 0.61)]),
        # (90, [(0.5, 0.6)]),
        # (90, [(0.5, 0.61)]),
        # (100, [(0.5, 0.6)]),
        # (110, [(0.5, 0.6)]),
        # (120, [(0.5, 0.6)]),
        # (120, [(0.5, 0.61)]),
        # (120, [(0.5, 0.62)]),
        # (120, [(0.5, 0.63)]),
        # (120, [(0.5, 0.64)]),
        # (120, [(0.5, 0.65)]),
        # (120, [(0.5, 0.66)]),
        # (120, [(0.5, 0.67)]),
        # (120, [(0.5, 0.68)]),
        # (120, [(0.5, 0.69)]),
        # (120, [(0.5, 0.7)]),
    ]
    sched = to_sched(*zip(*sched))
    # sched = to_sched_cont()

    def env_fn(): return TrailEnv(None)
    env = SubprocVecEnv([env_fn for _ in range(8)])
    eval_env = SubprocVecEnv([env_fn for _ in range(4)])

    inc_est_q_callback = EstimateQValCallback(sched=sched)
    adp_est_q_callback = EstimateQValCallback(sched=sched)

    discount = 0.975
    n_iters_per_ckpt = 3 * 1024
    tau = 0.95
    
    save_every=1
    cases = [
        # Case('Adaptive (Osc)', AdaptiveOscTeacher, {'conf':0.5}),

        # Case('Adaptive (Exp)', AdaptiveExpTeacher, teacher_params={'decision_point': 0.675, 'noise_range': 0.025, 'discount': discount}, cb_params={
        #     # 'next_lesson_callbacks': [adp_est_q_callback]
        # }),

        # Case('Incremental', IncrementalTeacher, teacher_params={'decision_point': 0.7, 'discount': discount}, cb_params={
        #     # 'next_lesson_callbacks': [inc_est_q_callback]
        # }),
        # Case('Random', RandomTeacher),
        # # Case('Final', FinalTaskTeacher),

        Case('Adaptive (Exp)', AdaptiveExpTeacher, teacher_params={'discount': discount, 'decision_point': 0.675, 'noise_range': 0.025, 'aggressive_checking': False}, cb_params={'save_every': save_every, 'save_path': f'{save_dir}/trained/adp/{run_id}'}),
        Case('Incremental', IncrementalTeacher, teacher_params={'discount': discount, 'decision_point': 0.7, 'aggressive_checking': False}, cb_params={'save_every': save_every, 'save_path': f'{save_dir}/trained/inc/{run_id}'}),
        Case('Random', RandomTeacher, cb_params={'save_every': save_every, 'save_path': f'{save_dir}/trained/rand/{run_id}'}),
    ]

    for i in tqdm(range(n_runs)):
        for case in cases:
            print('RUNNING', case.name)
            teacher = case.teacher(sched=sched, tau=tau, n_iters_per_ckpt=n_iters_per_ckpt, **case.teacher_params)
            model = make_model(env)
            # model = PPO.load('trained/osc_break/0/gen93')
            model.set_env(env)
            if 'save_path' in case.cb_params:
                case.cb_params['save_path'] += f'/{i}'

            traj = run_session(model, teacher, eval_env, case.cb_params, max_steps=2_000_000) 
            traj = [t[0] for t in traj]
            case.runs.append(traj)
        
    df = pd.DataFrame(cases)
    df.to_pickle(save_dir / f'meander_results_{run_id}.pkl')

    # inc_probs = np.array(inc_est_q_callback.probs)
    # np.save('meander_inc_probs.npy', inc_probs)

    # adp_probs = np.array(adp_est_q_callback.probs)
    # np.save('meander_adp_probs.npy', adp_probs)

# <codecell>
'''
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
# MANY LONG PLOTS

model_path = Path('trained/osc_break_ii/0/gen172.zip')

trail_args = {
    'width': 5,
    'diff_rate': 0.02,
    'radius': 70,
    'reward_dist': -1,
    'range': (-np.pi, np.pi)
}
trail_args['length'] = 500
trail_args['breaks'] = [(0.3, 0.32), (0.5, 0.53), (0.7, 0.72)]

model = PPO.load(model_path, device='cpu')
n_samps = 25

path = Path(f'trail_examples/')
if not path.exists():
    path.mkdir()

for i in tqdm(range(n_samps)):

    maps = []
    position_hists = []

    # print('preparing to generate headings')
    trail_map = MeanderTrail(**trail_args, heading=0)
    trail_map.max_steps = 1000

    env = TrailEnv(trail_map, discrete=True, treadmill=True)
    

    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, is_done, _ = env.step(action)

        if is_done:
            break

    # print('gen heading')
    maps.append(trail_map)
    position_hists.append(np.array(env.agent.position_history))

    fig, axs = plt.subplots(1, 1, figsize=(6, 12))

    for ax, m, p_hist in zip([axs], maps, position_hists):
        y_min = np.min(m.y_coords)
        y_max = np.max(m.y_coords)

        x_min = np.min(m.x_coords)
        x_max = np.max(m.x_coords)

        m.plot(ax=ax, xmin=x_min-20, xmax=x_max+20, ymin=y_min - 20, ymax=y_max + 20)
        ax.plot(p_hist[:,0], p_hist[:,1], linewidth=2, color='black')

    ratio = (y_max - y_min + 40) / (x_max - x_min + 40)
    height = 6 * ratio

    fig.set_size_inches((6, height))
    fig.tight_layout()

    plt.axis('off')
    plt.savefig(str(path / f'example_{i}.png'))
    # np.save(str(path / f'positions_{i}.npy'), p_hist)
    # with (path / f'map_{i}.pkl').open('wb') as fp:
    #     pickle.dump(m, fp)
    
    plt.clf()

# %%
'''