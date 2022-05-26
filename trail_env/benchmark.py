"""
Benchmark performance of various teacher strategies

author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
from tqdm import tqdm

from trail import TrailEnv
from trail_map import MeanderTrail
from curriculum import *


def make_model():
    return PPO("CnnPolicy", env, verbose=1,
                n_steps=128,
                batch_size=256,
                ent_coef=8e-6,
                gamma=0.98,
                gae_lambda=0.9,
                clip_range=0.3,
                max_grad_norm=1,
                vf_coef=0.36,
                n_epochs=16,
                learning_rate=0.0001,
                policy_kwargs={
                    'net_arch': [{'pi': [128, 32], 'vf': [128, 32]}],
                    'activation_fn': torch.nn.ReLU
                },
                device='auto'
                )

def run_session(student, teacher, eval_env):
    student.learn(total_timesteps=1000000, 
                  log_interval=5,
                  eval_env=eval_env, 
                  eval_freq=512, 
                  callback=[CurriculumCallback(teacher, eval_env)])
    return teacher.trajectory


if __name__ == '__main__':
    n_runs = 5
    len_sched = [10, 20, 30, 40, 50, 60]

    def env_fn(): return TrailEnv(None)
    env = SubprocVecEnv([env_fn for _ in range(8)])
    eval_env = env_fn()

    all_trajs_inc = []
    all_trajs_rand = []

    for _ in tqdm(range(n_runs)):
        teacher_inc = IncrementalTeacher(len_sched=len_sched)
        teacher_rand = RandomTeacher(TrailEnv, len_sched=len_sched)

        model_inc = make_model()
        model_rand = make_model()

        traj_inc = run_session(model_inc, teacher_inc, eval_env)
        traj_rand = run_session(model_rand, teacher_rand, eval_env)

        all_trajs_inc.append(traj_inc)
        all_trajs_rand.append(traj_rand)
    
    lens_inc = [len(traj) for traj in all_trajs_inc]
    lens_rand = [len(traj) for traj in all_trajs_rand]

    mean_inc = np.mean(lens_inc)
    std_inc = np.std(lens_inc)

    mean_rand = np.mean(lens_rand)
    std_rand = np.std(lens_rand)

    plt.bar([0, 1], [mean_inc, mean_rand], yerr=[2 * std_inc, 2 * std_rand], tick_label=['Incremental', 'Random'])
    plt.title('Teacher comparison on trail tracking task')
    plt.savefig('fig/tt_method_comparison.png')
    plt.clf()

    # plot trajectories
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    max_x = len(max(all_trajs_rand, key=lambda x: len(x)))

    for ax, trajs, name in zip(axs, (all_trajs_inc, all_trajs_rand), ('Incremental', 'Random')):
        ax_score = ax.twinx()
        for t in trajs:
            ns, score = zip(*t)
            ax.plot(ns, color='C0', alpha=0.7)
            ax_score.plot(score, color='C1', alpha=0.7)

        ax.set_xlabel('Lesson')
        ax.set_xlim(0, max_x)
        ax.set_ylabel('N', color='C0')
        ax_score.set_ylabel('Prob', color='C1')
        ax.set_title(name)

    fig.tight_layout()
    plt.savefig('fig/tt_trajs.png')

# %%
