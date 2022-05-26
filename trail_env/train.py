"""
Rough training code copied here
"""
# <codecell>
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from curriculum import *
from env import TrailEnv

# <codecell>
global_discrete = True
global_treadmill = True
trail_class = MeanderTrail
trail_args = {'narrow_factor': 5, 'length': 75, 'radius': 70, 'range': (-np.pi / 3, np.pi / 3)}

class SummaryCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.step_iter = 0

    def _on_step(self) -> bool:
        self.step_iter += 1
        return True

    def _on_training_end(self) -> None:
        print('TOTAL STEPS', self.num_timesteps)
        print('LOCAL STEPS', self.step_iter)


teacher = IncrementalTeacher(len_sched=[10, 20, 30, 40, 50, 60])
# teacher = RandomTeacher(TrailEnv)

def env_fn(): return TrailEnv(None, discrete=global_discrete, treadmill=global_treadmill)

# env = DummyVecEnv([env_fn for _ in range(8)])
env = SubprocVecEnv([env_fn for _ in range(8)])
eval_env = TrailEnv(None, discrete=global_discrete, treadmill=global_treadmill)

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

model.learn(total_timesteps=1000000, log_interval=5,
            eval_env=eval_env, eval_freq=512, callback=[CurriculumCallback(teacher, eval_env), SummaryCallback()])
model.save('trail_model')