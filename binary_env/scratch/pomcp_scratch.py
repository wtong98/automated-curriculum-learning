"""
Scratchwork when developing POMCP agent
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../')

from env import *

# <codecell>
N = 5
T = 50
student_lr = 0.002
p_eps = 0.1
L = 10
gamma = 0.95
lookahead_cap = 1
q_reinv_scale = 3   # Should be scaled adaptively?
q_reinv_prob = 0.25

es = np.zeros(N)

qrs_true = []

agent = TeacherPomcpAgent(goal_length=N, 
                          lookahead_cap=lookahead_cap, 
                          T=T, bins=L, p_eps=p_eps, student_qe=es, student_lr=student_lr, gamma=gamma, 
                          n_particles=1500, q_reinv_scale=q_reinv_scale, q_reinv_prob=q_reinv_prob)
env = CurriculumEnv(goal_length=N, train_iter=T, p_eps=p_eps, teacher_reward=10, student_reward=10, student_params={'lr': student_lr, 'q_e': es})

prev_obs = env.reset()
prev_a = None
is_done = False


while not is_done:
    a = agent.next_action(prev_a, prev_obs, with_replicas=0)

    state, reward, is_done, _ = env.step(a)
    obs = agent._to_bin(state[1])
    print(f'Took a: {a}   At n: {env.N}')

    assert agent.curr_n == env.N

    qrs_true.append([env.student.q_r[i] for i in range(N)])
    prev_a = a
    prev_obs = obs

qrs_true = qrs_true[1:]
print('Great success!')

# <codecell>
### PLOT RUN DIAGNOSTICS
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

steps = np.arange(len(agent.num_particles))

for i in range(N):
    axs[0].errorbar(steps, [q[i] for q in agent.qrs_means], yerr=[2 * s[i] for s in agent.qrs_stds], color=f'C{i}', alpha=0.5, fmt='o', markersize=0)
    axs[0].plot(steps, [q[i] for q in qrs_true[:-1]], label=f'qr[{i}]', color=f'C{i}', alpha=0.8)

axs[0].legend()
axs[0].set_xlabel('Step')
axs[0].set_ylabel('q')

axs[1].bar(steps, agent.num_particles)
axs[1].set_xlabel('Step')
axs[1].set_ylabel('# particles')

fig.suptitle('State estimates of POMCP agent')
fig.tight_layout()

plt.savefig('../fig/pomcp_state_estimate.png')

# <codecell>
### PLOT REPLICAS
# stds = np.std(agent.replicas, axis=1)
reps = np.array(agent.replicas)

counts = np.vstack((
    np.sum(reps == 0, axis=1),
    np.sum(reps == 1, axis=1),
    np.sum(reps == 2, axis=1))).T

freqs = counts / 5
entr = -freqs * np.log(freqs + 1e-8)
entr = np.sum(entr, axis=1)

plt.gcf().set_size_inches(8, 3)
plt.plot(entr)
plt.title('Entropy of actions across iterations')
plt.ylabel('Entropy')
plt.xlabel('Iteration')
plt.savefig('fig/pomcp_entropy_actions.png')

# %%
# <codecell>
### INVESTIGATE CLOSENESS OF MODEL TO REALITY
N = 10
T = 50
student_lr = 0.002
p_eps = 0.1
L = 10
es = np.zeros(N)

agent = TeacherPomcpAgent(goal_length=N, lookahead_cap=lookahead_cap, T=T, bins=10, p_eps=p_eps, student_qe=es, student_lr=student_lr)
env = CurriculumEnv(goal_length=N, train_iter=T, p_eps=p_eps, teacher_reward=10, student_reward=10, lr=student_lr, q_e=es)

iters = 100

all_pred_qrs = []
all_pred_obs = []
all_qrs = []
all_obs = []

env.reset()

for _ in range(iters):
    action = np.random.choice(3)

    q_r = env.student.q_r
    n = env.N
    ss = ()

    q_r = [q_r[i] for i in range(N)]
    (new_n, new_qr, ss), pred_obs, pred_reward, _ = agent._sample_transition((n, q_r, ss), action)

    state, reward, _, _ = env.step(action)

    # print('-T-R-U-E-')
    # print('QR', env.student.q_r)

    # assert state[0] == new_n       <-- violated by lookahead caps
    # assert reward == pred_reward   <-- violated by intermediate rewards

    all_pred_qrs.append(new_qr)
    all_pred_obs.append(pred_obs)
    all_qrs.append(env.student.q_r.copy())
    all_obs.append(agent._to_bin(state[1]))


# %%
plt.plot(all_pred_obs, '--o', label='Pred obs')
plt.plot(all_obs, '--o', label='True obs')
plt.yticks(np.arange(L+1))
plt.legend()

plt.title('Predicted vs. true observations')
plt.xlabel('Iteration')
plt.ylabel('Observation')
plt.savefig('fig/pomcp_debug_obs.png')

# %%
plt.plot([q[0] for q in all_pred_qrs], '--o', label='Pred qr[0]')
plt.plot([q[0] for q in all_qrs], '--o', label='Obs qr[0]')
plt.plot([q[1] for q in all_pred_qrs], '--o', label='Pred qr[1]', color='C0')
plt.plot([q[1] for q in all_qrs], '--o', label='Obs qr[1]', color='C1')
plt.legend()

plt.title('Predicted vs true states')
plt.xlabel('Iteration')
plt.ylabel('Q')
plt.savefig('fig/pomcp_debug_state.png')