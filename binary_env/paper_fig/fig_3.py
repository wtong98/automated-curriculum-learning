"""
POMCP and Adaptive Exp plots
"""

# <codecell>
import pickle

import sys
sys.path.append('../')

from env import *


def run_pomcp(n_iters=5000, eps=0, goal_length=3, T=3, gamma=0.9, lr=0.1, max_steps=500):
    global agent

    agent = TeacherPomcpAgentClean(goal_length=goal_length, 
                            T=T, bins=10, p_eps=0.05, gamma=gamma, 
                            n_particles=n_iters, q_reinv_prob=0.25)
    env = CurriculumEnv(goal_length=goal_length, train_iter=999, train_round=T, p_eps=0.05, teacher_reward=10, student_reward=10, student_qe_dist=eps, student_params={'lr': lr})
    traj = [env.N]
    all_qr = []
    prev_obs = env.reset()
    prev_a = None

    for _ in range(max_steps):
        a = agent.next_action(prev_a, prev_obs)

        state, _, is_done, _ = env.step(a)
        traj.append(env.N)

        qr = [env.student.q_r[i] for i in range(goal_length)]
        all_qr.append(qr)

        obs = agent._to_bin(state[1])
        prev_a = a
        prev_obs = obs

        if is_done:
            break
    
    return traj, {'qr': all_qr, 'teacher': agent}


def run_pomcp_with_retry(max_retries=5, max_steps=500, **kwargs):
    for i in range(max_retries):
        try:
            return run_pomcp(max_steps=max_steps, **kwargs)
        except Exception as e:
            print('pomcp failure', i+1)
            print(e)
    
    return [0] * max_steps, {}


n_iters = 1
results = []
for _ in range(n_iters):
    res = run_pomcp_with_retry(max_retries=5, eps=-2, goal_length=10, gamma=0.95)
    results.append(res)

with open('pomcp_res.pkl', 'wb') as fp:
    pickle.dump(results, fp)