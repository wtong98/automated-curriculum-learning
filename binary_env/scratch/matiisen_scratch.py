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

def run_incremental(eps=0, goal_length=3, T=3, max_steps=500, lr=0.1):
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
    
    return traj, {}


def run_random(eps=0, goal_length=3, T=3, max_steps=500, lr=0.1):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True)
    env.reset()
    traj = [env.N]

    for _ in range(max_steps):
        action = np.random.choice(goal_length) + 1
        _, _, is_done, _ = env.step(action)

        traj.append(env.N)

        if is_done:
            break
    
    return traj, {}


def run_final_task_only(eps=0, goal_length=3, T=3, max_steps=500, lr=0.1):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True)
    env.reset()
    traj = [env.N]

    for _ in range(max_steps):
        _, _, is_done, _ = env.step(goal_length)

        traj.append(env.N)

        if is_done:
            break
    
    return traj, {}


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
        xs[task_idx] = reward

        all_qs.append(qs.copy())
        traj.append(task_idx + 1)

        if is_done:
            break

    return traj, {'qs': all_qs}


def run_naive(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, alpha=0.1, beta=1, k=5):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True)
    env.reset()
    traj = [env.N]

    qs = np.zeros(goal_length)
    xs = np.zeros(goal_length)

    all_qs = [qs.copy()]

    for _ in range(max_steps // k):
        facs = np.exp(beta * np.abs(qs))
        probs = facs / np.sum(facs)

        task_idx = np.random.choice(goal_length, p=probs)
        all_scores = []

        for _ in range(k):
            (_, score), _, is_done, _ = env.step(task_idx + 1)
            all_scores.append(np.exp(score))
        res = linregress(range(k), all_scores)

        reward = res.slope - xs[task_idx]
        qs[task_idx] = alpha * reward + (1 - alpha) * qs[task_idx]
        xs[task_idx] = reward

        all_qs.append(qs.copy())
        traj.extend([task_idx + 1] * k)

        if is_done:
            break

    return traj, {'qs': all_qs}


def run_window(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, alpha=0.1, beta=1):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True)
    env.reset()
    traj = [env.N]

    qs = np.zeros(goal_length)
    xs = np.zeros(goal_length)

    all_scores = [[] for _ in range(goal_length)]
    all_times = [[] for _ in range(goal_length)]
    all_qs = [qs.copy()]

    for t in range(max_steps):
        facs = np.exp(beta * np.abs(qs))
        probs = facs / np.sum(facs)

        task_idx = np.random.choice(goal_length, p=probs)

        (_, score), _, is_done, _ = env.step(task_idx + 1)
        score = np.exp(score)
        all_scores[task_idx].append(score)
        all_times[task_idx].append(t)

        all_scores[task_idx] = all_scores[task_idx][-5:]
        all_times[task_idx] = all_times[task_idx][-5:]

        res = linregress(all_times[task_idx], all_scores[task_idx])

        slope = res.slope if not np.isnan(res.slope) else 0
        reward = slope - xs[task_idx]
        qs[task_idx] = alpha * reward + (1 - alpha) * qs[task_idx]
        xs[task_idx] = reward

        all_qs.append(qs.copy())
        traj.append(task_idx + 1)

        if is_done:
            break

    return traj, {'qs': all_qs}


def run_sampling(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, alpha=0.1):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True)
    env.reset()
    traj = [env.N]

    qs = np.zeros(goal_length)
    xs = np.zeros(goal_length)

    all_rewards = [[] for _ in range(goal_length)]
    all_qs = [qs.copy()]

    for _ in range(max_steps):
        reward_sample = [1 if len(buf) == 0 else np.random.choice(buf) for buf in all_rewards]
        task_idx = np.argmax(np.abs(reward_sample))

        (_, score), _, is_done, _ = env.step(task_idx + 1)
        score = np.exp(score)

        reward = score - xs[task_idx]
        qs[task_idx] = alpha * reward + (1 - alpha) * qs[task_idx]
        xs[task_idx] = reward

        all_rewards[task_idx].append(reward)
        all_rewards[task_idx] = all_rewards[task_idx][-5:]

        all_qs.append(qs.copy())
        traj.append(task_idx + 1)

        if is_done:
            break

    return traj, {'qs': all_qs}

N = 7
# traj, all_qs = run_naive(goal_length=N, eps=np.random.randn)

# qs = np.array(all_qs)
# for i, q in enumerate(qs.T):
#     plt.plot(q, label=f'task = {i+1}')

# plt.legend()

traj, _ = run_incremental(goal_length=N, eps=-1, max_steps=5000)
print(len(traj))
    



# %%


