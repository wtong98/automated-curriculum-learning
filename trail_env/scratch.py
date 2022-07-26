"""
Scratch experimentation
"""

# <codecell>
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch

from curriculum import *
from env import TrailEnv

# <codecell>
class TrackPerformanceCallback:
    def __init__(self) -> None:
        self.perfs = []
    
    def __call__(self, curriculum_cb: CurriculumCallback):
        t = curriculum_cb.teacher
        scores = []

        for s in t.sched:
            trail_map = t.trail_class(**s)
            test_env = TrailEnv(trail_map)
            scores.append(t._test_student(test_env))
        
        self.perfs.append(scores)

# <codecell>
if __name__ == '__main__':
    len_sched = 10 * (np.arange(10) + 1)
    teacher = IncrementalTeacher(len_sched=len_sched)

    def env_fn(): return TrailEnv(None)

    env = SubprocVecEnv([env_fn for _ in range(8)])
    # env = DummyVecEnv([env_fn for _ in range(8)])
    eval_env = TrailEnv(None)

    model = PPO("CnnPolicy", env, verbose=1,
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
                tensorboard_log='log',
                device='auto'
                )

    print('POLICY NETWORKS:')
    print(model.policy)
    # model.set_parameters('trained/epoch_3/possible_caster_feb5.zip')

    perf_cb = TrackPerformanceCallback()
    model.learn(total_timesteps=1000000, log_interval=5,
                eval_env=eval_env, eval_freq=512, callback=[CurriculumCallback(teacher, eval_env, next_lesson_callbacks=[perf_cb])])


# <codecell>
# TODO: visualize evolving epsilons
m = matplotlib.cm.get_cmap('viridis')

total = len(perf_cb.perfs)
for i, p in enumerate(perf_cb.perfs):
    plt.plot(p, '-', alpha=0.7, color=m(i/total))

plt.colorbar(matplotlib.cm.ScalarMappable(cmap=m))

# %%

# %%
total = len(perf_cb.perfs)
for i, p in enumerate(perf_cb.perfs):
    plt.plot(np.log(p), '-', alpha=0.7, color=m(i/total))

plt.colorbar(matplotlib.cm.ScalarMappable(cmap=m))

# %%
x = np.array(perf_cb.perfs)
# %%
