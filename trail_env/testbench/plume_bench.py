"""
Benchmarking teacher strategies on the plume tracking task. This script is
designed to be run with multiple parallel replications on a compute cluster.

author: William Tong (wtong@g.harvard.edu)
"""

from dataclasses import dataclass, field
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
from trail_map import *

def make_model(env, log_dir='log'):
    return PPO("CnnPolicy", env, verbose=1,
                n_steps=1024,
                batch_size=256,
                ent_coef=0.05,
                gamma=0.98,
                gae_lambda=0.9,
                clip_range=0.2,
                max_grad_norm=1,
                vf_coef=0.36,
                n_epochs=5,
                learning_rate=0.0001,
                tensorboard_log=log_dir,
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


def to_sched_range(rate_ranges):
    trail_args = {
        'wind_speed': 5,
        'length_scale': 20,
        'heading': 0
    }

    sched = [dict(start_rate_range=r, max_steps='auto', **trail_args) for r in rate_ranges]
    return sched


@dataclass
class Case:
    name: str = ''
    teacher: Callable = None
    teacher_handle: Teacher = None
    teacher_params: dict = field(default_factory=dict)
    cb_params: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)


if __name__ == '__main__':
    save_dir = Path('plume_runs')

    if not save_dir.exists():
        save_dir.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(None)
    run_id = rng.integers(999_999_999)
    print('RUN ID', run_id)

    n_runs = 1

    init_rate = 0.5
    rate_jump=0.75
    rate_spread = 0
    n_rates=4

    sec_rate = init_rate + n_rates * rate_jump
    sec_rate_jump=0.75
    n_rates2=4

    inv_rates = [init_rate + i * rate_jump for i in range(n_rates)] + [sec_rate + i * sec_rate_jump for i in range(n_rates2)]
    rates = [((1 / (r - rate_spread)), (1 / (r + rate_spread))) for r in inv_rates]
    sched = to_sched_range(rates)

    def env_fn(): return TrailEnv()

    env = SubprocVecEnv([env_fn for _ in range(8)])
    eval_env = SubprocVecEnv([env_fn for _ in range(4)])
    
    discount = 0.975
    n_iters_per_ckpt = 3 * 1024
    tau = 0.9

    save_every = 1
    cases = [
        Case('Adaptive', AdaptiveTeacher, teacher_params={'discount': discount, 'decision_point': 0.675, 'noise_range': 0.025, 'aggressive_checking': False}, cb_params={'save_every': save_every, 'save_path': f'{save_dir}/trained/adp/{run_id}'}),
        Case('Incremental', IncrementalTeacher, teacher_params={'discount': discount, 'decision_point': 0.7, 'aggressive_checking': False}, cb_params={'save_every': save_every, 'save_path': f'{save_dir}/trained/inc/{run_id}'}),
        Case('Random', RandomTeacher, cb_params={'save_every': save_every, 'save_path': f'{save_dir}/trained/rand/{run_id}'}),
    ]

    for i in tqdm(range(n_runs)):
        for case in cases:
            print('RUNNING', case.name)
            teacher = case.teacher(sched=sched, trail_class=PlumeTrail, tau=tau, n_iters_per_ckpt=n_iters_per_ckpt, **case.teacher_params)
            model = make_model(env, log_dir='log')
            model.set_env(env)

            traj = run_session(model, teacher, eval_env, case.cb_params, max_steps=3_000_000)
            case.runs.append(traj)

    df = pd.DataFrame(cases)

    filename = f'plume_results_{run_id}.pkl'
    df.to_pickle(save_dir / filename)
