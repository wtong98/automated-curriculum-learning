"""
Playing with algorithms from Matiisen et al.

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

import sys
sys.path.append('../')
from env import *

def run_incremental(eps=0, goal_length=3, T=3, max_steps=1000, lr=0.1):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr})
    env.reset()
    traj = [env.N]

    score = env._get_score(1, train=False)
    for _ in range(max_steps):
        if -score < env.p_eps:
            action = 2
        else:
            action = 1
        
        (_, score), _, is_done, _ = env.step(action)
        traj.append(env.N)

        if is_done:
            break
    
    return traj


def run_online(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, alpha=0.1, beta=1):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True)
    env.reset()
    traj = [env.N]

    qs = np.zeros(goal_length)
    xs = np.zeros(goal_length)

    all_qs = [qs.copy()]

    for _ in range(max_steps):
        facs = np.exp(beta * qs)
        probs = facs / np.sum(facs)

        task_idx = np.random.choice(goal_length, p=probs)
        (_, score), _, is_done, _ = env.step(task_idx + 1)

        reward = np.abs(np.exp(score) - xs[task_idx])
        qs[task_idx] = alpha * reward + (1 - alpha) * qs[task_idx]

        all_qs.append(qs.copy())
        traj.append(task_idx + 1)

        if is_done:
            break

    return traj, all_qs


def run_online(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, alpha=0.1, beta=1):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True)
    env.reset()
    traj = [env.N]

    qs = np.zeros(goal_length)
    xs = np.zeros(goal_length)

    all_qs = [qs.copy()]

    for _ in range(max_steps):
        facs = np.exp(beta * np.abs(qs))
        probs = facs / np.sum(facs)

        task_idx = np.random.choice(goal_length, p=probs)
        (_, score), _, is_done, _ = env.step(task_idx + 1)

        reward = np.exp(score) - xs[task_idx]
        qs[task_idx] = alpha * reward + (1 - alpha) * qs[task_idx]

        all_qs.append(qs.copy())
        traj.append(task_idx + 1)

        if is_done:
            break

    return traj, all_qs


def run_naive(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, alpha=0.1, beta=1, k=5):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True)
    env.reset()
    traj = [env.N]

    qs = np.zeros(goal_length)
    xs = np.zeros(goal_length)

    all_qs = [qs.copy()]

    for _ in range(max_steps):
        facs = np.exp(beta * np.abs(qs))
        probs = facs / np.sum(facs)

        task_idx = np.random.choice(goal_length, p=probs)
        all_scores = []

        for _ in range(k):
            (_, score), _, is_done, _ = env.step(task_idx + 1)
            all_scores.append(score)
        res = linregress(range(k), all_scores)

        reward = res.slope - xs[task_idx]
        qs[task_idx] = alpha * reward + (1 - alpha) * qs[task_idx]

        all_qs.append(qs.copy())
        traj.append(task_idx + 1)

        if is_done:
            break

    return traj, all_qs

# TODO: implement Window and Sampling algorithms


N = 15
traj, all_qs = run_naive(goal_length=N)

qs = np.array(all_qs)
for i, q in enumerate(qs.T):
    plt.plot(q, label=f'task = {i}')

plt.legend()

traj = run_incremental(goal_length=N)
print(len(traj))
    



# <codecell>
xs = np.linspace(-5, 5, num=100)
ys = sig(xs)
plt.plot(xs, ys)

