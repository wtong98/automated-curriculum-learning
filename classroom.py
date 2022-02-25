"""
Field for training agents and plotting results

author: William Tong (wtong@g.harvar.edu)
"""

# <codecell>
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from agent import Student, Teacher
from env import BinaryEnv, CurriculumEnv

def plot_path(path, completions):
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))

    axs[0].plot(path[:,0])
    for c in completions:
        if c == completions[0]:
            axs[0].axvline(x=c, color='red', alpha=0.3, label='Episode boundary')
        else:
            axs[0].axvline(x=c, color='red', alpha=0.3)

    axs[0].set_title('Teacher')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('N')
    axs[0].legend()

    axs[1].plot(path[:,1])
    for c in completions:
        if c == completions[0]:
            axs[1].axvline(x=c, color='red', alpha=0.3, label='Episode boundary')
        else:
            axs[1].axvline(x=c, color='red', alpha=0.3)

    axs[1].set_title('Student')
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('L')
    axs[1].legend()

    fig.tight_layout()
    # plt.savefig('fig/teacher_student_paths.png')

def plot_path_curve(path, completions):
    for i, c_start in enumerate(completions[:-1]):
        jitter = np.random.randn() * 0.4

        c_end = completions[i+1]
        curr_path = path[c_start+1:c_end]
        plt.plot(curr_path[:,1] + jitter, curr_path[:,0], '--o', alpha=0.8)

    
    plt.title('Common paths')
    plt.xlabel('L')
    plt.ylabel('N')



# <codecell> TEACHER TESTING
N = 10
T = 20
max_iters = 100000
eval_every = 1000
eval_len = 200

p_eps=0.1
teacher_reward=10
student_reward=10
qe_gen = lambda: np.random.normal(loc=0, scale=1)
qe_gen = None

def anneal_sched(i): 
    end_inv_temp = 10
    return (i / max_iters) * end_inv_temp

bins = 20
teacher = Teacher(bins=bins, anneal_sched=anneal_sched)

i = 0

path = []
global_completions = []

avg_time_to_comp = []
paths = []
comps = []
qs = []
def log(teacher):
    global i
    i += 1

    if i % eval_every == 0:
        eval_env = CurriculumEnv(N, T, 
            p_eps=p_eps, 
            student_reward=student_reward, teacher_reward=teacher_reward, 
            student_qe_dist=qe_gen)

        state = eval_env.reset()

        rewards = 0
        path = [teacher._to_bin(state)]
        completions = []
        for i in range(eval_len):
            a = teacher.next_action(state)
            state, reward, is_done, _ = eval_env.step(a)
            rewards += reward

            path.append(teacher._to_bin(state))
            if is_done:
                completions.append(i+1)
                state = eval_env.reset()

        total_time = completions[-1] if len(completions) > 0 else 0
        avg_time_to_comp.append(total_time / (len(completions) + 1e-8))
        paths.append(path)
        comps.append(completions)
        qs.append(teacher.q.copy())
        

def done(teacher):
    global i
    global_completions.append(i-1)

env = CurriculumEnv(N, T, 
    p_eps=p_eps, teacher_reward=teacher_reward, student_reward=student_reward, 
    student_qe_dist=qe_gen)
teacher.learn(env, max_iters=max_iters, use_tqdm=True, post_hook=log, done_hook=done)

path = np.array(path)
print('done!')


# <codecell>
# TODO: run experiment with heuristic and naive, compare results <-- STOPPED HERE
plt.plot(avg_time_to_comp, '--o')
plt.title('Average time to completion')
plt.xlabel('Time')
plt.ylabel('Reward')
plt.savefig('fig/teacher_average_ttc.png')

# %%
paths = np.array(paths)
comps = np.array(comps)

idx=-1
plot_path(paths[idx], comps[idx])
# plt.savefig('fig/teacher_path_converged.png')

# %%
def _make_heatmap(action_idx):
    im = np.zeros((N, 21))

    for i, row in enumerate(np.arange(N) + 1):
        for j, col in enumerate(np.arange(1, teacher.bins + 1)):
            im[i, j] = teacher.q[(row, col), action_idx]
    
    return im

ims = [_make_heatmap(i) for i in [0, 1, 2]]

fig, axs = plt.subplots(3, 1, figsize=(5, 6))
for i, (im, ax) in enumerate(zip(ims, axs.ravel())):
    m = ax.imshow(im, vmin=0, vmax=5)

    ax.set_title(f'Action: {i-1}')
    ax.set_xlabel('L')
    ax.set_ylabel('N')
    fig.colorbar(m, ax=ax)

fig.suptitle('Teacher Q values')
fig.tight_layout()
plt.savefig('fig/teacher_q.png')


# %%
### PLOT PHASE DIAGRAM OF STRATEGY
ns = np.arange(N) + 1
ls = np.arange(0, teacher.bins + 1)

ll, nn = np.meshgrid(ls, ns)
actions = []

for l, n in zip(ll.ravel(), nn.ravel()):
    a = teacher.next_action((n, l), is_binned=True)  # TODO: actually already a bin state (deal only on bin states)
    actions.append(a)

z = np.array(actions).reshape(ll.shape) - 1
# plt.contourf(ll, nn, z)
plt.imshow(z)
plt.colorbar()

plt.xlabel('L')
plt.ylabel('N')
plt.title('Sampled action')

plt.savefig('fig/teacher_actions.png')

# %%
### ENTROPY OF STRAT
entropy = []

for l, n in zip(ll.ravel(), nn.ravel()):
    probs = teacher.policy((n, l))
    entropy.append(-np.sum(probs * np.log(probs)))

z = np.array(entropy).reshape(ll.shape)
# plt.contourf(ll, nn, z)
plt.imshow(z)
plt.colorbar()

plt.xlabel('L')
plt.ylabel('N')
plt.title('Policy entropy')

plt.savefig('fig/teacher_entropy.png')


# %% VARIOUS PATHS THAT STUDENTS TAKE
paths = np.array(paths)
comps = np.array(comps)

idx=-1
plot_path_curve(paths[idx], comps[idx])
plt.savefig('fig/teacher_sample_paths.png')

# %% ESTIMATE PHASE DIAGRAM FROM STUDENT TRAJECTORIES
eval_len = 100000
eval_env = CurriculumEnv(N, T, 
    p_eps=p_eps, 
    student_reward=student_reward, teacher_reward=teacher_reward, 
    student_qe_dist=None)

state = eval_env.reset()
transitions = []

for i in tqdm(range(eval_len)):
    a = teacher.next_action(state)
    transitions.append((teacher._to_bin(state), a))
    state, reward, is_done, _ = eval_env.step(a)

    if is_done:
        state = eval_env.reset()

# %%
est_a = []
counts = defaultdict(int)
for l in np.arange(1, 21):
    for (s, a) in transitions:
        if s[1] == l:
            counts[l, a] += 1

for l, n in zip(ll.ravel(), nn.ravel()):
    raw_counts = np.array([counts[l,a] for a in [0, 1, 2]])
    if np.sum(raw_counts) == 0:
        a = -2
    else:
        probs = raw_counts / np.sum(raw_counts)
        a = np.argmax(probs) - 1

    # est_a.append(-np.sum(probs * np.log(probs)))
    est_a.append(a)

z = np.array(est_a).reshape(ll.shape)
# plt.contourf(ll, nn, z)
plt.imshow(z)
plt.colorbar()
plt.savefig('fig/teacher_actions_smoothed.png')

# <codecell>
est_a = []
for l, n in zip(ll.ravel(), nn.ravel()):
    raw_counts = np.array([counts[l,a] for a in [0, 1, 2]])
    if np.sum(raw_counts) == 0:
        probs = [1/3, 1/3, 1/3]
        a = -2
    else:
        probs = raw_counts / np.sum(raw_counts)
        a = np.argmax(probs) - 1

    est_a.append(-np.sum(probs * np.log(np.array(probs) + 1e-6)))
    if np.isnan(est_a[-1]):
        print('probs', probs)

z = np.array(est_a).reshape(ll.shape)
# plt.contourf(ll, nn, z)
plt.imshow(z)
plt.colorbar()
plt.savefig('fig/teacher_actions_smoothed_entropy.png')