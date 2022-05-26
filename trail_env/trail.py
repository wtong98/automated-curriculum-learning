"""
Trail-tracking environment

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from stable_baselines3 import PPO
from skimage.transform import rescale
from typing import Tuple

import gym
from gym import spaces

try:
    from trail_map import *
    from curriculum import *
except ImportError as e:
    from .trail_map import *
    from .curriculum import *
except Exception as e:
    raise e


class TrailEnv(gym.Env):
    heading_bound = np.pi
    max_speed = 3
    view_distance = 40
    max_steps = 100
    max_off_trail_steps = np.inf
    observation_scale = 2
    y_adjust = 0.3

    def __init__(self, trail_map=None, discrete=True, treadmill=True):
        super().__init__()

        self.discrete = discrete
        self.treadmill = treadmill

        self.next_map = None
        if trail_map == None:
            trail_map = MeanderTrail(width=20, length=35, diff_rate=0.01, radius=100, reward_dist=3, range=(-np.pi / 4, np.pi / 4))

        """
        The action space is the tuple (heading, velocity).
            Heading refers to the change in direction the agent is heading,
                in radians. North is equivalent to heading=0
            Velocity is the step-size the agent progresses in the environment
        """

        if self.discrete:
            self.action_space = spaces.Discrete(8)
            if self.treadmill:
                # self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
                self.action_space = spaces.Discrete(3)

        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

        """
        Observe the strength of odor in an ego-centric frame of reference. This
        space can be interpreted as a 2 * view_distance x 2 * view_distance x num_channels
        images.
            The first channel represents the agent's location history
            The second channel represents the agent's odor history
        """
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(2 * TrailEnv.view_distance * TrailEnv.observation_scale, 2 * TrailEnv.view_distance * TrailEnv.observation_scale, 3),
                                            dtype=np.uint8)

        self.map = trail_map
        self.agent = TrailAgent(self.map, TrailEnv.view_distance, scale=TrailEnv.observation_scale, y_adjust=TrailEnv.y_adjust)
        self.curr_step = 0

    def step(self, action):
        if self.discrete:
            if self.treadmill:
                d_action = action - 1
                d_heading = d_action * np.pi / 4
                self.agent.move(d_heading, TrailEnv.max_speed)
            else:
                heading = (action / self.action_space.n) * 2 * np.pi
                self.agent.move_abs(heading, TrailEnv.max_speed)
        else:
            if self.treadmill:
                heading = action[0] * np.pi / 4  # 'tuned' to /4
                speed = ((action[1] + 1) / 1) * TrailEnv.max_speed # 'tuned' to /1
                self.agent.move(heading, speed)
            else:
                heading = action[0] * np.pi
                speed = ((action[1] + 1) / 2) * TrailEnv.max_speed
                self.agent.move_abs(heading, speed)

        self.agent.sniff()

        obs = self.agent.make_observation()
        reward, is_done, is_success = self.agent.get_reward()

        if self.curr_step == TrailEnv.max_steps:
            is_done = True

        self.curr_step += 1

        return obs, reward, is_done, {'is_success': is_success}

    def reset(self):
        self.curr_step = 0
        if self.next_map != None:
            print('SWITCHING TO MAP:', self.next_map)
            self.map = self.next_map
            self.next_map = None
        else:
            self.map.reset()

        self.agent = TrailAgent(self.map, TrailEnv.view_distance, scale=TrailEnv.observation_scale, y_adjust=TrailEnv.y_adjust)
        obs = self.agent.make_observation()
        return obs

    def queue_map(self, next_map):
        self.next_map = next_map


class TrailAgent:
    def __init__(self, trail_map, view_distance, scale=1, y_adjust=0):
        self.position = [0, 0]
        self.heading = 0
        self.map = trail_map
        self.view_distance = view_distance
        self.observation_scale = scale
        self.y_adjust = y_adjust

        self.position_history = [[0, 0]]
        self.odor_history = []
        self.off_trail_step = 0

        self.sniff()

    def move(self, d_heading, speed):
        self.heading += d_heading
        dx = np.sin(self.heading) * speed
        dy = np.cos(self.heading) * speed

        self.position[0] += dx
        self.position[1] += dy
        self.position_history.append(self.position[:])

    def move_direct(self, dx, dy):
        self.position[0] += dx
        self.position[1] += dy
        self.position_history.append(self.position[:])

    def move_abs(self, heading, speed):
        dx = np.sin(heading) * speed
        dy = np.cos(heading) * speed

        self.position[0] += dx
        self.position[1] += dy
        self.position_history.append(self.position[:])

    def sniff(self):
        odor = self.map.sample(*self.position)
        self.odor_history.append((odor, *self.position[:]))
        return odor

    def get_reward(self) -> Tuple[float, bool]:
        # reward = 10 * (self.odor_history[-1][0] - self.odor_history[-2][0])
        reward = -2
        is_success = False

        if np.isclose(self.map.sample(*self.position), 0, atol=1e-2):
            self.off_trail_step += 1
            if self.off_trail_step == TrailEnv.max_off_trail_steps:
                return reward, True
        else:
            self.off_trail_step = 0

        if self.map.is_at_checkpoint(*self.position):
            reward = 5

        is_done = self.map.is_done(*self.position)
        if is_done:
            reward = 100
            is_success = True

        return reward, is_done, is_success

    def make_observation(self):
        pos_obs = self.make_pos_observation()
        odor_obs = self.make_odor_observation()

        self_obs = np.zeros((2 * self.view_distance * self.observation_scale, 2 * self.view_distance * self.observation_scale))
        total_obs = np.stack((pos_obs, odor_obs, self_obs), axis=-1)
        return total_obs

    def make_pos_observation(self):
        pos_img = np.zeros((2 * self.view_distance, 2 * self.view_distance))
        past_pos = np.vstack(self.position_history)

        orig_trans = -np.tile(self.position, (len(self.position_history), 1))
        rot_ang = self.heading
        rot_trans = np.array([
            [np.cos(rot_ang), -np.sin(rot_ang)],
            [np.sin(rot_ang), np.cos(rot_ang)]
        ])

        ego = (past_pos + orig_trans) @ rot_trans.T
        ego_pos = ego + self.view_distance
        ego_pos[:,1] += int(self.y_adjust * self.view_distance)  # shift upwards

        # Manhattan interpolation
        for i, point in enumerate(ego_pos[:-1]):
            next_point = ego_pos[i + 1]
            d = next_point - point
            steps = 2 * np.sum(np.abs(next_point - point)).astype('int')
            dx = (d[0] / steps) if steps != 0 else 0
            dy = (d[1] / steps) if steps != 0 else 0

            for i in range(steps):
                x_coord = np.round(point[0] + i * dx).astype(int)
                y_coord = np.round(point[1] + i * dy).astype(int)

                # for x_ in range(x_coord - 1, x_coord + 2):
                #     for y_ in range(y_coord - 1, y_coord + 2):
                #         if 0 <= x_ < self.view_distance * 2 \
                #                 and 0 <= y_ < self.view_distance * 2:
                #             pos_img[x_, y_] = 255
                if 0 <= x_coord < self.view_distance * 2 \
                        and 0 <= y_coord < self.view_distance * 2:
                    pos_img[x_coord, y_coord] = 255

        pos_img = np.flip(pos_img.T, axis=0)
        pos_img_scale = rescale(pos_img, self.observation_scale)
        return pos_img_scale.astype(np.uint8)

    def make_odor_observation(self):
        odor_img = np.zeros((2 * self.view_distance, 2 * self.view_distance))
        past = np.vstack(self.odor_history)
        past_odor = past[:, 0]
        past_pos = past[:, 1:]

        orig_trans = -np.tile(self.position, (len(past_pos), 1))
        rot_ang = self.heading
        rot_trans = np.array([
            [np.cos(rot_ang), -np.sin(rot_ang)],
            [np.sin(rot_ang), np.cos(rot_ang)]
        ])

        ego = (past_pos + orig_trans) @ rot_trans.T
        ego_pos = ego + self.view_distance
        ego_pos[:,1] += int(self.y_adjust * self.view_distance)  # shift upwards

        for odor, pos in zip(past_odor, ego_pos):
            x_pos = int(pos[0])
            y_pos = int(pos[1])
            for x_ in range(x_pos - 1, x_pos + 2):
                for y_ in range(y_pos - 1, y_pos + 2):
                    x_coord, y_coord = x_, y_
                    if 0 <= x_coord < self.view_distance * 2 - 1 \
                            and 0 <= y_coord < self.view_distance * 2 - 1:

                        x = np.round(x_coord).astype(int)
                        y = np.round(y_coord).astype(int)
                        odor_img[x, y] = odor * 255

        odor_img = np.flip(odor_img.T, axis=0)
        odor_img_scale = rescale(odor_img, self.observation_scale)
        return odor_img_scale.astype(np.uint8)

    def render(self):
        self.map.plot()
        plt.plot(*self.position, 'ro')
        plt.show()


class TrailContinuousEnv(TrailEnv):

    def __init__(self, trail_map=None):
        super().__init__(trail_map=trail_map, discrete=False, treadmill=True)


global_discrete = True
global_treadmill = True
trail_class = MeanderTrail
trail_args = {'narrow_factor': 5, 'length': 75, 'radius': 70, 'range': (-np.pi / 3, np.pi / 3)}

# <codecell>

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

if __name__ == '__main__':
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
    exit()
    
'''
# <codecell>
# env = TrailEnv(trail_class(**trail_args))
# model = PPO.load('trail_model')

# env.play_anim(model)

# <codecell>
# env = TrailEnv(trail_class(**trail_args), discrete=global_discrete)
trail_map = StraightTrail(end=np.array([-15, 0]), narrow_factor=3)
trail_map.tol = 4
env = TrailEnv(trail_map, discrete=global_discrete)
# model = PPO.load('trail_model')
model = PPO.load('trail_model.zip')

obs = env.reset()
plt.imshow(obs)
for _ in range(20):
    action, _ = model.predict(obs, deterministic=True)
    print(action)
    obs, reward, is_done, _ = env.step(action)
    print(is_done)
    print('last reward:', reward)

    plt.imshow(obs)
    plt.show()

    if is_done:
        break

env.map.plot()

# TODO: test for longer and longer trails, especially where
# odor signals become saturated
print(env.agent.odor_history)

# <codecell>
# env = TrailEnv(trail_class(), discrete=global_discrete)
env = TrailEnv(StraightTrail(end=np.array([-15, 0]), narrow_factor=3), discrete=global_discrete)
model = PPO.load('trail_model')

obs = env.reset()
frames = [obs]
plt.imshow(obs)
for _ in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, is_done, _ = env.step(action)
    frames.append(obs)

    if is_done:
        print('reach end')
        break

ani = FuncAnimation(plt.gcf(), lambda f: plt.imshow(f), frames=frames)
ani.save('out.gif')


# <codecell>
env = TrailEnv(discrete=True, treadmill=True)

obs = env.reset()
plt.imshow(obs[..., 0])
plt.show()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, _, _ = env.step(action)
    print('action', action)
    plt.imshow(obs)
    plt.show()


# <codecell>
env = TrailEnv()
check_env(env)
print('done!')

# <codecell>

trail_map = StraightTrail()
agent = TrailAgent(trail_map, view_distance=25)
# print(agent.get_reward())
agent.move(0, 5)
agent.sniff()
print(agent.get_reward())
agent.move(np.pi / 4, 5)
odor = agent.sniff()
print(agent.get_reward())
agent.move(-np.pi / 2, 5)
agent.sniff()
print(agent.get_reward())
print(agent.odor_history)
# agent.move(-np.pi / 4, 0)
# agent.move(-np.pi / 6, 2)

obs = agent.make_observation()

img_corr = np.flip(obs.T, axis=0)
# print(img_corr)
plt.imshow(obs)
print('ODOR', agent.odor_history)

# <codecell>
env = TrailEnv(StraightTrail(end=np.array([15, 15])), discrete=True)
env.step(4)
env.step(4)
env.step(2)
env.step(4)
env.step(4)

obs, _, _, _ = env.step(4)
print('heading:', env.agent.heading)

print(env.agent.position_history)

plt.imshow(obs)

# <codecell>
env = TrailEnv(StraightTrail(end=np.array([15, 15])), discrete=False)
env.step([0, 3])
# env.step([np.pi / 4, 3])

obs, _, _, _ = env.step([np.pi / 6, 3])
print('heading:', env.agent.heading)

print(env.agent.position_history)

plt.imshow(obs)
# %%
# trail_map = trail_class(is_eval=True, narrow_factor=3)
trail_map = trail_class(**trail_args)
trail_map.plot()

# %%
'''
