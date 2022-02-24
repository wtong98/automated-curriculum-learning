"""
Field for training agents and plotting results

author: William Tong (wtong@g.harvar.edu)
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np

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
    axs[1].set_ylabel('Log prob')
    axs[1].legend()

    fig.tight_layout()
    # plt.savefig('fig/teacher_student_paths.png')

# <codecell> TEACHER TESTING
N = 10
T = 20
max_iters = 100000
eval_every = 1000
eval_len = 100

p_eps=0.1
teacher_reward=100
student_reward=100
qe_gen = lambda: np.random.normal(loc=0, scale=0.5)
qe_gen = None

def anneal_sched(i): 
    end_inv_temp = 2
    return (i / max_iters) * end_inv_temp

bins = 20
teacher = Teacher(bins=bins, anneal_sched=anneal_sched)

i = 0

path = []
global_completions = []

avg_rewards = []
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
        path = [state]
        completions = []
        for i in range(eval_len):
            a = teacher.next_action(state)
            state, reward, is_done, _ = eval_env.step(a)
            rewards += reward

            path.append(state)
            if is_done:
                completions.append(i+1)
                state = eval_env.reset()

        avg_rewards.append(rewards / eval_len)
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
plt.plot([anneal_sched(i) for i in range(max_iters)])

# <codecell>
plt.plot(avg_rewards, '--o')

# %%
paths = np.array(paths)
comps = np.array(comps)

idx=-1
plot_path(paths[idx], comps[idx])

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
    ax.set_xlabel('Probability of success')
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

# %%
### ENTROPY OF STRAT
entropy = []

for l, n in zip(ll.ravel(), nn.ravel()):
    probs = teacher.policy((n, l))
    entropy.append(-np.log(np.max(probs)))

z = np.array(entropy).reshape(ll.shape)
# plt.contourf(ll, nn, z)
plt.imshow(z)
plt.colorbar()

# TODO: report patchy results to Gautam <-- STOPPED HERE
# (agent prefers to alternate rather than stay still)