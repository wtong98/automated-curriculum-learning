"""
Benchmarking teacher strategies on the trail tracking task. This script is
designed to be run with multiple parallel replications on a compute cluster.

author: William Tong (wtong@g.harvard.edu)
"""

from dataclasses import dataclass, field
from pathlib import Path

from typing import Callable
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

def logit(x):
    return np.log(x / (1 - x))

@dataclass
class Case:
    name: str = ''
    teacher: Callable = None
    teacher_params: dict = field(default_factory=dict)
    cb_params: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)

if __name__ == '__main__':
    save_dir = Path('trail_runs')

    if not save_dir.exists():
        save_dir.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(None)
    run_id = rng.integers(999_999_999)
    print('RUN ID', run_id)

    n_runs = 1
    sched = [
        (10, [(0.5, 0.6)]),
        (30, [(0.5, 0.6)]),
        (50, [(0.5, 0.6)]),
        (70, [(0.5, 0.6)]),
        (90, [(0.5, 0.6)]),
        (100, [(0.5, 0.63)]),
    ]

    sched = to_sched(*zip(*sched))

    def env_fn(): return TrailEnv(None)
    env = SubprocVecEnv([env_fn for _ in range(8)])
    eval_env = SubprocVecEnv([env_fn for _ in range(4)])

    discount = 0.975
    n_iters_per_ckpt = 3 * 1024
    tau = 0.95
    
    save_every=1
    cases = [
        Case('Adaptive', AdaptiveTeacher, teacher_params={'discount': discount, 'decision_point': 0.675, 'noise_range': 0.025, 'aggressive_checking': False}, cb_params={'save_every': save_every, 'save_path': f'{save_dir}/trained/adp/{run_id}'}),
        Case('Incremental', IncrementalTeacher, teacher_params={'discount': discount, 'decision_point': 0.7, 'aggressive_checking': False}, cb_params={'save_every': save_every, 'save_path': f'{save_dir}/trained/inc/{run_id}'}),
        Case('Random', RandomTeacher, cb_params={'save_every': save_every, 'save_path': f'{save_dir}/trained/rand/{run_id}'}),
    ]

    for i in tqdm(range(n_runs)):
        for case in cases:
            print('RUNNING', case.name)
            teacher = case.teacher(sched=sched, tau=tau, n_iters_per_ckpt=n_iters_per_ckpt, **case.teacher_params)
            model = make_model(env)
            model.set_env(env)
            if 'save_path' in case.cb_params:
                case.cb_params['save_path'] += f'/{i}'

            traj = run_session(model, teacher, eval_env, case.cb_params, max_steps=2_000_000) 
            case.runs.append(traj)
        
    df = pd.DataFrame(cases)
    df.to_pickle(save_dir / f'meander_results_{run_id}.pkl')
