"""
Common utilities
"""
import shutil

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

import sys
sys.path.append('../../')

from env import TrailEnv

def load_model(model_path):
    model = PPO.load(model_path, device='cpu', custom_objects={'lr_schedule': 0, 'clip_range': 0})
    return model

def run_model(model_path, trail_map):
    model = PPO.load(model_path, device='cpu', custom_objects={'lr_schedule': 0, 'clip_range': 0})
    env = TrailEnv(trail_map, discrete=True, treadmill=True)

    obs = env.reset()
    all_obs = [obs]
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, is_done, _ = env.step(action)

        all_obs.append(obs)

        if is_done:
            break

    return np.array(env.agent.position_history), {'agent': env.agent, 'obs': all_obs}


def plot_run(trail_map, pos_hist, save_path=None, ax=None, **plot_kwargs):
    if ax is None:
        ax = plt.gca()

    trail_map.plot(ax=ax, **plot_kwargs)
    ax.plot(pos_hist[:,0], pos_hist[:,1], linewidth=2, color='black')

    # fig.tight_layout()
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path)


def plot_observations(obs, fig_path):
    obs_dir = fig_path / f'example_{i}_obs'
    if obs_dir.exists():
        shutil.rmtree(obs_dir)
    
    obs_dir.mkdir()
    for i, obs in enumerate(obs):
        plt.clf()
        plt.gca().set_axis_off()
        plt.imshow(obs / (np.max(obs, axis=(0, 1)) + 1e-8), interpolation='nearest')
        plt.savefig(obs_dir / f'obs_{i}.png')