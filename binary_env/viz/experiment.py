"""
All experiments for generating trajectories

author: William Tong (wtong@g.harvard.edu)
"""
from dataclasses import dataclass, field
import sys
from typing import Callable

from scipy.stats import linregress
from tqdm import tqdm

sys.path.append('../')
from env import *

@dataclass
class Case:
    name: str = ''
    run_func: Callable = None
    run_params: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)
    info: list = field(default_factory=list)

class NormalDist:
    def __init__(self, loc=0, scale=1) -> None:
        self.loc = loc
        self.scale = scale
    
    def __call__(self):
        return np.random.normal(loc=self.loc, scale=self.scale)

def run_exp(n_iters, cases, use_tqdm=False, **global_kwargs):
    gen = range(n_iters)
    if use_tqdm:
        gen = tqdm(gen)

    for _ in gen:
        for case in cases:
            traj, info = case.run_func(**case.run_params, **global_kwargs)
            case.runs.append(traj)
            case.info.append(info)


def run_incremental(eps=0, goal_length=3, T=3, max_steps=500, lr=0.1):
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr})
    env.reset()
    traj = [env.N]
    all_qr = []

    score = env._get_score(1, train=False)
    for _ in range(max_steps):
        if -score < env.p_eps:
            action = 2
        else:
            action = 1
        
        (_, score), _, is_done, _ = env.step(action)

        traj.append(env.N)
        
        qr = [env.student.q_r[i] for i in range(goal_length)]
        all_qr.append(qr)

        if is_done:
            break
    
    return traj, {'qr': all_qr}


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
        reward = slope - xs[task_idx]
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


def run_adp_osc(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, **teacher_kwargs):
    teacher = TeacherUncertainOsc(goal_length, **teacher_kwargs)
    env = CurriculumEnv(goal_length=goal_length, student_reward=10, student_qe_dist=eps, train_iter=999, train_round=T, student_params={'lr': lr}, anarchy_mode=True, return_transcript=True)
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
    
    return traj, {'qr': all_qr}


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

def run_pomcp(n_iters=5000, eps=0, goal_length=3, T=3, gamma=0.9, lr=0.1, max_steps=500):
    global agent

    agent = TeacherPomcpAgentClean(goal_length=goal_length, 
                            T=T, bins=10, p_eps=0.05, gamma=gamma, 
                            n_particles=n_iters, q_reinv_prob=0.25)
    env = CurriculumEnv(goal_length=goal_length, train_iter=999, train_round=T, p_eps=0.05, teacher_reward=10, student_reward=10, student_qe_dist=eps, student_params={'lr': lr})
    traj = [env.N]
    prev_obs = env.reset()
    prev_a = None

    for _ in range(max_steps):
        a = agent.next_action(prev_a, prev_obs)

        state, _, is_done, _ = env.step(a)
        traj.append(env.N)

        obs = agent._to_bin(state[1])
        prev_a = a
        prev_obs = obs

        if is_done:
            break
    
    return traj, {}


def run_pomcp_with_retry(max_retries=5, max_steps=500, **kwargs):
    for i in range(max_retries):
        try:
            return run_pomcp(max_steps=max_steps, **kwargs)
        except Exception as e:
            print('pomcp failure', i+1)
            print(e)
    
    return [0] * max_steps, {}


"""Continuous algorithms"""
def to_cont(N=3, eps=0, dn_per_interval=100):
    prob = sig(eps) ** (1 / dn_per_interval)
    eps_cont = np.log(prob / (1 - prob))
    N_cont = N * dn_per_interval

    return N_cont, eps_cont

def run_incremental_perfect(eps=0, T=3, goal_length=3, n_step=100, lr=0.1, max_steps=1000, threshold=0.8):
    N, e = to_cont(N=goal_length, eps=eps)
    env = CurriculumEnv(goal_length=N, student_reward=10, student_qe_dist=e, train_round=T, student_params={'lr': lr, 'n_step': n_step}, anarchy_mode=True)
    env.reset()
    traj = [env.N]
    all_qr = []

    for _ in range(max_steps):
        qs = np.array([e + env.student.q_r[i] for i in range(N)])
        while len(qs) > 0 and np.exp(np.sum(np.log(sig(qs)))) < threshold:
            qs = qs[:-1]
        
        action = min(len(qs) + 1, N)
        _, _, is_done, _ = env.step(action)
        traj.append(env.N)
        
        qr = qs - e
        all_qr.append(qr)

        if is_done:
            break
    
    return traj, {'qr': all_qr}


def run_adp_cont(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, conf=0.2, tau=0.5, **kwargs):
    N, e = to_cont(N=goal_length, eps=eps)
    teacher = TeacherAdaptive(N, conf=conf, tau=tau, **kwargs)
    env = CurriculumEnv(goal_length=N, student_reward=10, student_qe_dist=e, train_round=T, student_params={'lr': lr, 'n_step': 100}, anarchy_mode=True, return_transcript=True)
    traj = [env.N]
    env.reset()
    all_qr = []

    obs = (1, [])
    for _ in range(max_steps):
        action = teacher.next_action(obs)
        obs, _, is_done, _ = env.step(action)
        traj.append(env.N)

        qr = np.array([env.student.q_r[i] for i in range(N)])
        all_qr.append(qr)

        if is_done:
            break
    
    return traj, {'qr': all_qr}


def run_adp_exp_cont(eps=0, goal_length=3, T=3, lr=0.1, max_steps=500, **kwargs):
    N, e = to_cont(N=goal_length, eps=eps)

    splits = np.array([0.7, 0])
    dec_to_idx = np.array([3, 7, 0, 2])
    tree = TeacherTree(splits)
    teacher = TeacherExpAdaptive(N, tree, dec_to_idx, **kwargs)
    env = CurriculumEnv(goal_length=N, student_reward=10, student_qe_dist=e, train_round=T, student_params={'lr': lr, 'n_step': 100}, anarchy_mode=True, return_transcript=True)
    traj = [env.N]
    env.reset()
    all_qr = []

    obs = (1, [])
    for _ in range(max_steps):
        action = teacher.next_action(obs)
        obs, _, is_done, _ = env.step(action)
        traj.append(env.N)

        qr = np.array([env.student.q_r[i] for i in range(N)])
        all_qr.append(qr)

        if is_done:
            break
    
    return traj, {'qr': all_qr}