"""
Common utilities
"""
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

import sys
sys.path.append('../../')

from env import TrailEnv

def run_model(model_path, trail_map):
    model = PPO.load(model_path, device='cpu')
    env = TrailEnv(trail_map, discrete=True, treadmill=True)

    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, is_done, _ = env.step(action)

        if is_done:
            break

    return np.array(env.agent.position_history)


def plot_run(trail_map, pos_hist, save_path=None):
    ax = plt.gca()

    trail_map.plot(ax=ax)
    ax.plot(pos_hist[:,0], pos_hist[:,1], linewidth=2, color='black')

    # fig.tight_layout()
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path)
