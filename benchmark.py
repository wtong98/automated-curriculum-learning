"""
Probe the performance of the teacher on various benchmarks,
and compared to other training systems
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from tqdm import tqdm

from env import BinaryEnv, Student, Teacher, CurriculumEnv

class NoTeacherTest:
    def __init__(self, goal_length, k=1):
        self.goal_length = goal_length
        self.k = k
    
    def run(self, student, T, max_iters=1000, student_reward=1):
        self.iter = 0

        for _ in range(max_iters):
            for _ in range(self.k):
                student.learn(BinaryEnv(self.goal_length, reward=student_reward), max_iters=T)
            self.iter += 1

            final_score = student.score(self.goal_length)
            if np.isclose(final_score, 0, atol=1e-1):
                break
        
        return self.iter


class NaiveTest:
    def __init__(self, goal_length, k=1):
        self.goal_length = goal_length
        self.k = k
    
    def run(self, student, T, max_iters=1000, student_reward=1):
        self.iter = 0

        for _ in range(max_iters):
            n = np.random.choice(self.goal_length) + 1
            for _ in range(self.k):
                student.learn(BinaryEnv(n, reward=student_reward), max_iters=T)
            self.iter += 1

            final_score = student.score(self.goal_length)
            if np.isclose(final_score, 0, atol=1e-1):
                break
        
        return self.iter


class IncrementalTest:
    def __init__(self, goal_length, k=1):
        self.goal_length = goal_length
        self.k = k
    
    def run(self, student, T, max_iters=1000, student_reward=1):
        self.iter = 0

        n = 1
        all_steps = [n]
        for _ in range(max_iters):
            for _ in range(self.k):
                student.learn(BinaryEnv(n, reward=student_reward), max_iters=T)
            
            curr_score = student.score(n)
            all_steps.append(n)
            if np.isclose(curr_score, 0, atol=1e-1):
                if n == self.goal_length:
                    break
                else:
                    n += 1

            self.iter += 1

        return self.iter, all_steps


class TeacherHeuristicTest:
    def __init__(self, goal_length, lr=0.1, k=5) -> None:
        self.goal_length = goal_length
        self.q = np.zeros(goal_length)
        self.lr = lr
        self.k = k

    # softmax policy
    def policy(self) -> np.ndarray:
        probs = np.exp(np.abs(self.q)) / np.sum(np.exp(np.abs(self.q)))
        return probs
    
    def next_action(self):
        probs = self.policy()
        return np.random.choice(len(probs), p=probs)

    def update(self, state, reward):
        self.q[state] = self.lr * reward + (1 - self.lr) * self.q[state]
    
    def reset(self):
        self.q[:] = 0
    
    def run(self, student, T, max_iters=1000, student_reward=1, scale=100):
        self.iter = 0
        all_scores = []
        all_probs = []
        all_steps = []

        for _ in range(max_iters):
            N = self.next_action() + 1
            all_steps.append(N)

            scores = []
            for _ in range(self.k):
                # student.learn(BinaryEnv(N, reward=student_reward), max_iters=T, 
                #             post_hook=lambda s: scores.append(s.score(N)))
                total_comp = 0
                success_comp = 0
                def done(student, reward):
                    nonlocal total_comp, success_comp
                    if reward > 0:
                        success_comp += 1
                    total_comp += 1

                student.learn(BinaryEnv(N, reward=student_reward), max_iters=T,
                              done_hook=done)
                
                ratio = success_comp / total_comp if total_comp > 0 else 0
                scores.append(ratio)
            
            slope, _, _, _, _ = linregress(np.arange(len(scores)), scores)
            slope *= scale
            self.update(N - 1, slope)
            self.iter += 1

            scores = [student.score(n) for n in (np.arange(len(self.q)) + 1)]
            all_scores.append(scores)
            all_probs.append(self.policy())

            final_score = student.score(len(self.q))
            if np.isclose(final_score, 0, atol=1e-1):
                break
        
        return self.iter, all_scores, all_probs, all_steps


class TeacherAgentTest:
    def __init__(self, teacher_agent, goal_length, k=1):
        self.teacher = teacher_agent
        self.goal_length = goal_length
        self.k = k
    
    def run(self, student, T, max_iters=1000, student_reward=1):
        self.iter = 0
        N = 1

        all_steps = [N]
        for _ in range(max_iters):
            for _ in range(self.k):
                student.learn(BinaryEnv(N, reward=student_reward), max_iters=T)

            log_p = student.score(N)
            a = self.teacher.next_action((N, log_p)) - 1
            N = np.clip(N + a, 1, self.goal_length)
            all_steps.append(N)

            self.iter += 1

            final_score = student.score(self.goal_length)
            if np.isclose(final_score, 0, atol=1e-1):
                break
        
        return self.iter, all_steps


def train_teacher(N=10, T=20, bins=20, p_eps=0.1,
                  teacher_reward=10, teacher_gamma=1, student_reward=10,
                  qe_gen=None, anneal_sched=None,
                  max_iters=100000, eval_every=1000, eval_len=200):

    teacher = Teacher(bins=bins, anneal_sched=anneal_sched, gamma=teacher_gamma, lr=0.1)

    i = 0
    path = []

    avg_time_to_comp = []
    paths = []
    comps = []
    qs = []

    def log(teacher):
        nonlocal i
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
                if is_done and reward > 0:
                    completions.append(i+1)
                    state = eval_env.reset()

            total_time = completions[-1] if len(completions) > 0 else 0
            avg_time_to_comp.append(total_time / (len(completions) + 1e-8))
            paths.append(path)
            comps.append(completions)
            qs.append(teacher.q.copy())
            
    env = CurriculumEnv(N, T, 
        p_eps=p_eps, teacher_reward=teacher_reward, student_reward=student_reward, 
        student_qe_dist=qe_gen)
    teacher.learn(env, max_iters=max_iters, use_tqdm=True, post_hook=log)

    path = np.array(path)
    print('done!')
    return {
        'teacher': teacher,
        'avg_time_to_comp': avg_time_to_comp,
        'paths': paths,
        'comps': comps,
        'qs': qs
    }

# <codecell>
### TRAIN TEACHER AGENT(S)
max_iters = 100000
# qe_gen = lambda: np.random.normal(loc=0, scale=0.5)
qe_gen = None

N = 10
T = 20

teacher_reward = 10
student_reward = 10

def anneal_sched(i): 
    end_inv_temp = 10
    return (i / max_iters) * end_inv_temp

results = train_teacher(N=N, T=T, bins=30, p_eps=0.05, teacher_gamma=1, max_iters=max_iters, anneal_sched=anneal_sched, qe_gen=qe_gen,
    student_reward=student_reward, teacher_reward=teacher_reward, eval_len=1500, eval_every=5000)

# <codecell>
### SANITY CHECK PLOT
plt.plot(results['avg_time_to_comp'], '--o')
plt.title('Average time to completion')
plt.xlabel('Time')
plt.ylabel('Iterations')
plt.savefig('fig/teacher_average_ttc.png')

# <codecell>
### GATHER PERFORMANCE METRICS
iters = 50
K = 5

naive_scores = []
inc_scores = []
heuristic_scale_100_scores = []
heuristic_no_scale_scores = []
agent_scores = []

naive_test = NaiveTest(N, k=1)
inc_test = IncrementalTest(N, k=1)
heuristic_test = TeacherHeuristicTest(N, k=5)
agent_test = TeacherAgentTest(results['teacher'], N, k=1)

for _ in tqdm(range(iters)):
    naive_scores.append(
        naive_test.run(Student(), T, max_iters=10000, student_reward=student_reward))

    inc_scores.append(
        inc_test.run(Student(), T, max_iters=10000, student_reward=student_reward)[0])

    heuristic_no_scale_scores.append(heuristic_test.run(
        Student(), T, max_iters=10000, student_reward=student_reward, scale=1)[0])
    
    heuristic_test.reset()

    heuristic_scale_100_scores.append(
        heuristic_test.run(Student(), T, max_iters=10000, student_reward=student_reward, scale=100)[0])

    agent_scores.append(agent_test.run(
        Student(), T, max_iters=10000, student_reward=student_reward)[0])

# %%
all_scores = [
    naive_scores, 
    inc_scores,
    # heuristic_no_scale_scores, 
    # heuristic_scale_100_scores, 
    agent_scores]

labels = [
    'Random teacher',
    'Incremental',
    # 'Heuristic, scale=1',
    # 'Heuristic, scale=100',
    'Teacher agent'
]

plt.gcf().set_size_inches(6, 4)
plt.title('Average number of iterations to train a student (k = 5)')
plt.ylabel('Iterations')

all_means = [np.mean(score) for score in all_scores]
all_se = [2 * np.std(score) / np.sqrt(iters) for score in all_scores]

plt.bar(np.arange(len(all_scores)), height=all_means, yerr=all_se, tick_label=labels)
plt.savefig('fig/acl_method_comparison_only_teachers_trimmed.png')


# <codecell>
### COMPARE PERFORMANCE FOR PARAMS OF TEACHER AGENT
bins = [3, 5, 7, 10, 15, 20, 30]
all_results = []

max_iters = 80000
qe_gen = None

N = 10
T = 20

teacher_reward = 10
student_reward = 10

def anneal_sched(i): 
    end_inv_temp = 10
    return (i / max_iters) * end_inv_temp

for b in bins:
    results = train_teacher(N=N, T=T, bins=b, max_iters=max_iters, anneal_sched=anneal_sched, qe_gen=qe_gen,
        student_reward=student_reward, teacher_reward=teacher_reward)
    all_results.append(results)

# %%
iters = 50
all_scores = []

for results in tqdm(all_results):
    test = TeacherAgentTest(results['teacher'], N)
    scores = []
    for _ in range(iters):
        scores.append(test.run(
            Student(), T, max_iters=10000, student_reward=student_reward))
    all_scores.append(scores)

# <codecell>
all_scores = np.array(all_scores)
all_means = np.mean(all_scores, axis=1)
all_se = 2 * np.std(all_scores, axis=1) / np.sqrt(iters)


plt.bar(np.arange(len(all_scores)), height=all_means, yerr=all_se, tick_label=bins)
plt.title('Number of iterations to train student for varying bin sizes')
plt.xlabel('Number of bins')
plt.ylabel('Iterations')
plt.savefig('fig/acl_method_comparison_varying_bins.png')


# <codecell>
##### HEURISTIC DIAGNOSTICS
N = 10
T = 20
student_reward=10

all_steps_heuristic = []

for _ in range(5):
    test = TeacherHeuristicTest(N, k=5)
    iters, scores, probs, steps_heur = test.run(
        Student(), T, max_iters=10000, student_reward=student_reward, scale=1)
    
    all_steps_heuristic.append(steps_heur)

scores = np.array(scores).T
probs = np.array(probs).T

# <codecell>
fig, axs = plt.subplots(1, 2, figsize=(11, 4))

for i, (score, prob) in enumerate(zip(scores, probs)):
    axs[0].plot(score, label=f'N={i+1}')
    axs[0].set_title('Score')

    axs[1].plot(prob, label=f'N={i+1}')
    axs[1].set_title('Probability of selecting task')

axs[0].legend()
axs[1].legend()

fig.suptitle('Learning curves for each task (scale=1)')
fig.tight_layout()

plt.savefig('fig/acl_heuristic_learning_curves.png')


# <codecell>
####### COMPARING STEPS
all_steps_agent = []
all_steps_inc = []

student = None
for _ in range(5):
    curr_student = Student()
    _, steps_agent = TeacherAgentTest(results['teacher'], 10, k=1).run(curr_student, 20, max_iters=1000, student_reward=10)
    if len(steps_agent) > 500:
        print('err')
        student = curr_student
    _, steps_inc = IncrementalTest(10, k=1).run(Student(), 20, max_iters=1000, student_reward=10)

    all_steps_agent.append(steps_agent)
    all_steps_inc.append(steps_inc)

# <codecell>
for i, steps in enumerate(all_steps_inc):
    if i == 0:
        plt.plot(steps, label='Incremental', alpha=0.6, color='C0')
    else:
        plt.plot(steps, alpha=0.6, color='C0')

for i, steps in enumerate(all_steps_agent):
    # if len(steps) > 500:
    #     steps = steps[:100]
    if i == 0:
        plt.plot(steps, label='Agent', alpha=0.6, color='C1')
    else:
        plt.plot(steps, alpha=0.6, color='C1')



plt.title('Steps taken')
plt.xlabel('Iteration')
plt.ylabel('N')
plt.legend()
plt.savefig('fig/acl_steps_taken.png')
# %%

'''