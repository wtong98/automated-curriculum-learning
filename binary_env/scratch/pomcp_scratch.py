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
N = 3
T = 5
# student_lr = 0.002
student_lr = 0.1
p_eps = 0.05
L = 10
gamma = 0.99
lookahead_cap = 1
q_reinv_scale = 3   # Should be scaled adaptively?
q_reinv_prob = 0

es = np.ones(N) * -5

qrs_true = []

agent = TeacherPomcpAgent(goal_length=N, 
                          lookahead_cap=lookahead_cap, 
                          T=T, bins=L, p_eps=p_eps, student_qe=es, student_lr=student_lr, gamma=gamma, 
                          n_particles=10000, q_reinv_scale=q_reinv_scale, q_reinv_prob=q_reinv_prob)
env = CurriculumEnv(goal_length=N, train_iter=999, train_round=T, p_eps=p_eps, teacher_reward=10, student_reward=10, student_qe_dist=es, student_params={'lr': student_lr})

prev_obs = env.reset()
prev_a = None
is_done = False


# TODO: q estimates seem to be way off
while not is_done:
    a = agent.next_action(prev_a, prev_obs, with_replicas=0)

    state, reward, is_done, _ = env.step(a)
    obs = agent._to_bin(state[1])
    print('---')
    print(f'Took a: {a}   At n: {env.N}')

    assert agent.curr_n == env.N

    qrs_true.append([env.student.q_r[i] for i in range(N)])
    prev_a = a
    prev_obs = obs

# qrs_true = qrs_true[1:]
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

# plt.savefig('../fig/pomcp_state_estimate.png')

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
# plt.savefig('fig/pomcp_entropy_actions.png')

# %%
# <codecell>
### INVESTIGATE CLOSENESS OF MODEL TO REALITY
N = 10
T = 5
student_lr = 0.1
p_eps = 0.1
L = 10
es = np.ones(N) * -1
lookahead_cap = 1

agent = TeacherPomcpAgent(goal_length=N, lookahead_cap=lookahead_cap, T=T, bins=10, p_eps=p_eps, student_qe=es, student_lr=student_lr)
env = CurriculumEnv(goal_length=N, train_iter=999, train_round=T, p_eps=p_eps, teacher_reward=10, student_reward=10, student_qe_dist=es, student_params={'lr': student_lr})

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
    (new_n, new_qr), pred_obs, pred_reward, _ = agent._sample_transition((n, q_r), action)

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
# plt.savefig('fig/pomcp_debug_obs.png')

# %%
plt.plot([q[0] for q in all_pred_qrs], '--o', label='Pred qr[0]')
plt.plot([q[0] for q in all_qrs], '--o', label='Obs qr[0]')
plt.plot([q[1] for q in all_pred_qrs], '--o', label='Pred qr[1]', color='C0')
plt.plot([q[1] for q in all_qrs], '--o', label='Obs qr[1]', color='C1')
plt.legend()

plt.title('Predicted vs true states')
plt.xlabel('Iteration')
plt.ylabel('Q')
# plt.savefig('fig/pomcp_debug_state.png')


# %% SYSTEMATICALLY INVESTIGATE TRANSITION MODEL
N = 3
T = 5
student_lr = 0.1
p_eps = 0.1
L = 10
es = np.ones(N) * -1
lookahead_cap = 1

agent = TeacherPomcpAgent(goal_length=N, lookahead_cap=lookahead_cap, T=T, bins=10, p_eps=p_eps, student_qe=es, student_lr=student_lr)
env = CurriculumEnv(goal_length=N, train_iter=999, train_round=T, p_eps=p_eps, teacher_reward=10, student_reward=10, student_qe_dist=es, student_params={'lr': student_lr})

iters = 1000
qr_start = [5.58, 4.7, 1]
n_start = 3
action = 1


all_pred_qr = []
all_true_qr = []

for _ in range(iters):
    env.reset()
    env.student.q_r = qr_start[:]
    env.N = n_start

    (new_n, pred_qr), pred_obs, pred_reward, _ = agent._sample_transition((n_start, qr_start[:]), action)
    # env.step(action)
    # true_qr = [env.student.q_r[i] for i in range(N)]
    student = env.student
    student.learn(BinaryEnv(n_start, reward=10), max_iters=999, max_rounds=T)
    true_qr = student.q_r 

    all_pred_qr.append(pred_qr)
    all_true_qr.append(true_qr)

all_pred_qr = np.array(all_pred_qr)
all_true_qr = np.array(all_true_qr)
pred_means = np.mean(all_pred_qr, axis=0)
true_means = np.mean(all_true_qr, axis=0)

print('PRED_MEANS', pred_means)
print('TRUE_MEANS', true_means)

plt.hist(all_true_qr[:,2], bins=20, alpha=0.7)
plt.hist(all_pred_qr[:,2], bins=20, alpha=0.7)

qs = all_true_qr + es
log_p = np.sum(np.log(agent._sig(pred_means + es)))
# log_p = np.sum([-np.log(1 + np.exp(-q)) for q in (pred_means + agent.student_qe)[:3]])
agent._to_bin(log_p)
# %%

pred_qr = agent._update_qr(n_start+1, qr_start[:], fail_idx=3)

env.reset()
env.train_round = 1
env.student.q_r = qr_start[:]
env.N = n_start
env.step(2)

true_qr = [env.student.q_r[i] for i in range(3)]

print(pred_qr)
print(true_qr)
# %%
