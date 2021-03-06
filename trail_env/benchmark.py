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
                n_steps=512,
                batch_size=256,
                ent_coef=0.01,
                gamma=0.975,
                gae_lambda=0.9,
                clip_range=0.2,  # TODO: try using sched?
                max_grad_norm=1,
                vf_coef=0.36,
                n_epochs=10,
                learning_rate=0.0001,
                tensorboard_log='log',
                policy_kwargs={
                    'net_arch': [{'pi': [128, 32], 'vf': [128, 32]}],
                    'activation_fn': torch.nn.ReLU
                },
                device='auto'
                )

def run_session(student, teacher, eval_env, cb_params):
    student.learn(total_timesteps=500000, 
                  eval_env=eval_env, 
                  eval_freq=512, 
                  callback=[CurriculumCallback(teacher, eval_env=eval_env, **cb_params)])
    return teacher.trajectory


Case = namedtuple('Case', ['name', 'teacher', 'teacher_params', 'cb_params', 'traj'])

if __name__ == '__main__':
    n_runs = 1
    len_sched = [10, 20, 30, 40, 50, 60, 70, 80]
    # len_sched = [10, 20, 30, 40, 50, 60]

    def env_fn(): return TrailEnv(None)
    env = SubprocVecEnv([env_fn for _ in range(8)])
    eval_env = env_fn()

    cases = [
        # Case('Incremental', IncrementalTeacher, {'len_sched': len_sched}, []),
        Case('Oscillator', OscillatingTeacher, {'len_sched': len_sched}, {'save_every': 1, 'save_path': 'trained/osc'}, []),
        # Case('Naive', NaiveTeacher, {'len_sched': len_sched}, [])
    ]

    for i in tqdm(range(n_runs)):
        for case in cases:
            teacher = case.teacher(**case.teacher_params)
            model = make_model(env)
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
    axs[0].set_yticks(np.arange(len(len_sched)))

    # axs[0].set_xlim((800, 900))

    all_lens = [[len(run) for run in case.traj] for case in cases]
    all_means = [np.mean(lens) for lens in all_lens]
    all_serr = [2 * np.std(lens) / np.sqrt(n_runs) for lens in all_lens]
    all_names = [case.name for case in cases]

    axs[1].bar(np.arange(len(cases)), all_means, tick_label=all_names, yerr=all_serr)
    axs[1].set_ylabel('Iterations')

    fig.suptitle(f'Trail sched = {len_sched}')
    fig.tight_layout()
    plt.savefig('trained/osc/0/tt_trajs.png')

# # <codecell>
#     lens_inc = [len(traj) for traj in all_trajs_inc]
#     lens_rand = [len(traj) for traj in all_trajs_rand]

#     mean_inc = np.mean(lens_inc)
#     std_inc = np.std(lens_inc) / np.sqrt(n_runs)

#     mean_rand = np.mean(lens_rand)
#     std_rand = np.std(lens_rand) / np.sqrt(n_runs)

#     plt.bar([0, 1], [mean_inc, mean_rand], yerr=[2 * std_inc, 2 * std_rand], tick_label=['Incremental', 'Random'])
#     plt.title('Teacher comparison on trail tracking task')
#     plt.savefig('fig/tt_method_comparison.png')
#     plt.clf()
# # <codecell>
#     # plot trajectories
#     fig, axs = plt.subplots(2, 1, figsize=(12, 6))
#     max_x = len(max(all_trajs_rand, key=lambda x: len(x)))

#     for ax, trajs, name in zip(axs, (all_trajs_inc, all_trajs_rand), ('Incremental', 'Random')):
#         ax_score = ax.twinx()
#         for t in trajs:
#             ns, score = zip(*t)
#             ax.plot(ns, 'o--', color='C0', alpha=0.6)
#             ax_score.plot(score, 'o--', color='C1', alpha=0.6)

#         ax.set_xlabel('Lesson')
#         ax.set_xlim(0, max_x)
#         ax.set_ylabel('N', color='C0')
#         ax_score.set_ylabel('Prob', color='C1')
#         ax.set_title(name)

#     fig.tight_layout()
#     plt.savefig('fig/tt_trajs.png')

# %%  SHOWCASE PERFORMANCE IN PLOTS
save_path = Path('trained/osc/0')
max_gen = 10

trail_args = {
    'length': 80,
    'width': 5,
    'diff_rate': 0.01,
    'radius': 100,
    'reward_dist': -1,
    'range': (-np.pi, np.pi)
}

for i in tqdm(range(1, max_gen + 1)):
    model_path = save_path / f'gen{i}'
    # print('loading model')
    model = PPO.load(model_path, device='cuda')

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

