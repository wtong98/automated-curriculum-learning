"""
Common utils useful for making these plots
"""

# <codecell>
from dataclasses import dataclass, field
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

import sys
sys.path.append('../')
from env import *

@dataclass
class Case:
    name: str = ''
    run_func: Callable = None
    run_params: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)
    info: list = field(default_factory=list)


def run_exp(n_iters, cases, use_tqdm=False, **global_kwargs):
    gen = range(n_iters)
    if use_tqdm:
        gen = tqdm(gen)

    for _ in gen:
        for case in cases:
            traj, info = case.run_func(**case.run_params, **global_kwargs)
            case.runs.append(traj)
            case.info.append(info)


def plot_traj_and_qr(traj, qr, eps, N, n_step=1, ax=None, save_path=None):
    if type(ax) == type(None):
        plt.clf()
        plt.gcf().set_size_inches(8, 3)
        ax = plt.gca()

    qr = np.array(qr)
    qr = np.flip(qr.T, axis=0) + eps
    im = ax.imshow(qr, aspect='auto', vmin=0, vmax=10)

    ticks = np.arange(N) * n_step
    ax.set_yticks(ticks, np.flip(np.arange(N) + 1))
    ax.set_ylabel('N')
    ax.set_xlabel('Steps')
    ax.set_title(fr'$\epsilon = {eps}$')

    plt.colorbar(im, ax=ax)

    adj = -0.425 if n_step == 1 else 2.75
    ax.plot(10 * n_step - np.array(traj)[1:] + adj, color='red')
    ax.set_xlim((0, len(traj) - 1.5))

    plt.gcf().tight_layout()

    if save_path:
        plt.savefig(save_path)

def plot_traj_slices(qr, ax, eps, n_steps=1):
    qr = np.array(qr) + eps

    for i in [0, 2, 4, 7, 9]:
        ax.plot(qr[:,i*n_steps + n_steps - 1], label=f'N = {i+1}', alpha=i/15 + 0.35, color='C0')

    ax.set_xlabel('Step')
    ax.set_ylabel(r'Q value')
    ax.legend()

def run_exp_inc(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, **teacher_kwargs):
    teacher = TeacherExpIncremental(**teacher_kwargs)
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, return_transcript=True)
    traj = [env.N]
    env.reset()
    all_qr = []

    obs = (1, [])
    for _ in range(max_steps):
        action = teacher.next_action(obs)
        obs, _, is_done, _ = env.step(action)
        traj.append(env.N)

        qr = [env.student.q_r[i] for i in range(goal_length)]
        all_qr.append(qr)

        if is_done:
            break
    
    return traj, {'qr': all_qr, 'teacher': teacher}


def run_pomcp(n_iters=5000, eps=0, goal_length=3, T=3, gamma=0.9, lr=0.1, max_steps=500):
    global agent

    agent = TeacherPomcpAgentClean(goal_length=goal_length, 
                            T=T, bins=10, p_eps=0.05, gamma=gamma, 
                            n_particles=n_iters, q_reinv_prob=0.25)
    env = CurriculumEnv(goal_length=goal_length, train_iter=999, train_round=T, p_eps=0.05, teacher_reward=10, student_reward=10, student_qe_dist=eps, student_params={'lr': lr}, return_transcript=True)
    traj = [env.N]
    all_qr = []
    prev_obs = env.reset()
    prev_a = None

    for _ in range(max_steps):
        a = agent.next_action(prev_a, prev_obs)
        print('ACTION', a)

        state, _, is_done, _ = env.step(a)
        traj.append(env.N)

        qr = [env.student.q_r[i] for i in range(goal_length)]
        all_qr.append(qr)

        obs = tuple(state[1])
        prev_a = a
        prev_obs = obs

        if is_done:
            break
    
    return traj, {'qr': all_qr, 'teacher': agent}


# <codecell>
def plot_pomcp_diagnostics(info, eps):
    t = info['teacher']
    for i, (qs_true, qs, std) in enumerate(
        zip(np.array(info['qr']).T, np.array(t.qrs_means).T, np.array(t.qrs_stds).T)):

        xs = np.arange(len(qs))
        plt.fill_between(xs, qs - 2 * std, qs + 2 * std, alpha=0.2, color=f'C{i}')
        plt.plot(qs_true[:-1], color=f'C{i}', label=f'q_{i+1}')

    lr_means = np.array(t.lr_means)
    lr_stds = np.array(t.lr_stds)
    plt.fill_between(xs, lr_means - 2 * lr_stds, lr_means + 2 * lr_stds, alpha=0.2, color='black')
    plt.plot(xs, len(xs) * [0.1], color='black', label='lr')

    qes_means = np.array(t.qes_means)
    qes_stds = np.array(t.qes_stds)
    plt.fill_between(xs, qes_means - 2 * qes_stds, qes_means + 2 * qes_stds, color='red', alpha=0.2)
    plt.plot(xs, len(xs) * [eps], color='red', label='eps')

    plt.legend()
# <codecell>


def run_pomcp_with_retry(max_retries=5, max_steps=500, **kwargs):
    for i in range(max_retries):
        try:
            return run_pomcp(max_steps=max_steps, **kwargs)
        except Exception as e:
            print('pomcp failure', i+1)
            print(e)
    
    return [0] * max_steps, {}


def run_adp_exp_disc(eps=0, goal_length=3, T=3, max_steps=500, lr=0.1):
    splits = [0.8, 0]   # TODO: optimize
    dec_to_idx = [0, 1, 0, 2]
    tree = TeacherTree(splits)
    teacher = TeacherExpAdaptive(goal_length, tree, dec_to_idx, discrete=True)

    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True, return_transcript=True)
    traj = [env.N]
    all_qr = []
    env.reset()

    obs = (1, [])
    for _ in range(max_steps):
        action = teacher.next_action(obs)
        obs, _, is_done, _ = env.step(action)
        traj.append(env.N)

        qr = [env.student.q_r[i] for i in range(goal_length)]
        all_qr.append(qr)

        if is_done:
            break

    return traj, {'qr': all_qr}


def run_random(eps=0, goal_length=3, T=3, max_steps=500, lr=0.1, is_cont=False):
    if is_cont:
        goal_length, eps = to_cont(goal_length, eps)
    
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


def run_final_task_only(eps=0, goal_length=3, T=3, max_steps=500, lr=0.1, is_cont=False):
    if is_cont:
        goal_length, eps = to_cont(goal_length, eps)

    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True)
    env.reset()
    env.N = goal_length
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

        # reward = res.slope - xs[task_idx]
        reward = res.slope
        qs[task_idx] = alpha * reward + (1 - alpha) * qs[task_idx]
        xs[task_idx] = reward

        all_qs.append(qs.copy())
        traj.extend([task_idx + 1] * k)

        if is_done:
            break

    return traj, {'qs': all_qs}


def run_window(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, alpha=0.1, beta=1, k=5):
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

        all_scores[task_idx] = all_scores[task_idx][-k:]
        all_times[task_idx] = all_times[task_idx][-k:]

        res = linregress(all_times[task_idx], all_scores[task_idx])

        slope = res.slope if not np.isnan(res.slope) else 0
        # reward = slope - xs[task_idx]
        reward = slope
        qs[task_idx] = alpha * reward + (1 - alpha) * qs[task_idx]
        xs[task_idx] = reward

        all_qs.append(qs.copy())
        traj.append(task_idx + 1)

        if is_done:
            break

    return traj, {'qs': all_qs}


def run_sampling(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, alpha=0.1, k=5):
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
        all_rewards[task_idx] = all_rewards[task_idx][-k:]

        all_qs.append(qs.copy())
        traj.append(task_idx + 1)

        if is_done:
            break

    return traj, {'qs': all_qs}


"""Continuous algorithms"""
def to_cont(N=3, eps=0, dn_per_interval=100):
    prob = sig(eps) ** (1 / dn_per_interval)
    eps_cont = np.log(prob / (1 - prob))
    N_cont = N * dn_per_interval

    return N_cont, eps_cont

# TODO: tune
def run_exp_cont(eps=0, goal_length=3, n_step=100, T=3, lr=0.1, max_steps=500, **kwargs):
    N, e = to_cont(N=goal_length, eps=eps, dn_per_interval=n_step)

    splits = np.array([0.7, 0])
    dec_to_idx = np.array([3, 7, 0, 2])
    tree = TeacherTree(splits)
    teacher = TeacherExpAdaptive(N, tree, dec_to_idx, **kwargs)

    env = CurriculumEnv(goal_length=N, student_reward=10, student_qe_dist=e, train_round=T, student_params={'lr': lr, 'n_step': 100}, anarchy_mode=True, return_transcript=True)
    env.reset()
    env.N = n_step
    traj = [env.N]
    all_qr = []

    obs = (n_step, [])
    for _ in range(max_steps):
        action = teacher.next_action(obs)
        obs, _, is_done, _ = env.step(action)
        traj.append(env.N)

        qr = np.array([env.student.q_r[i] for i in range(N)])
        all_qr.append(qr)

        if is_done:
            break
    
    return traj, {'qr': all_qr}

def run_inc_cont(eps=0, T=3, goal_length=3, n_step=100, lr=0.1, max_steps=500, **teacher_kwargs):
    N, e = to_cont(N=goal_length, eps=eps, dn_per_interval=n_step)

    teacher = TeacherExpIncremental(**teacher_kwargs)
    env = CurriculumEnv(goal_length=N, student_reward=10, student_qe_dist=e, train_round=T, student_params={'lr': lr, 'n_step': n_step}, anarchy_mode=True, return_transcript=True)
    env.reset()
    env.N = n_step
    traj = [env.N]
    all_qr = []

    obs = (n_step, [])
    for _ in range(max_steps):
        a = teacher.next_action(obs)
        diff = (a - 1) * n_step
        action = env.N + diff
        
        obs, _, is_done, _ = env.step(action)
        traj.append(env.N)

        qr = [env.student.q_r[i] for i in range(N)]
        all_qr.append(qr)

        if is_done:
            break
    
    return traj, {'qr': all_qr, 'teacher': teacher}


# traj, info = run_inc_cont(goal_length=10, eps=2)
# plt.plot(traj)

# traj, info = run_exp_cont(goal_length=10, eps=-2)
# plt.plot(traj)