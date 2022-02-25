"""
Simple binary environments for experimenting with teacher-student interactions

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from collections import defaultdict
import gym
import numpy as np

from scipy.stats import linregress
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class BinaryEnv(gym.Env):
    def __init__(self, length, reward=1) -> None:
        super().__init__()
        self.length = length

        self.observation_space = gym.spaces.Discrete(length + 1)
        self.action_space = gym.spaces.Discrete(2)

        self.reward = reward
        self.loc = 0
    
    def step(self, action):
        reward = 0
        is_done = False

        if action == 0:
            is_done = True
        else:
            self.loc += action
            if self.loc == self.length:
                reward = self.reward
                is_done = True
            
        return self.loc, reward, is_done, {}
    
    def reset(self):
        self.loc = 0
        return 0


class CurriculumEnv(gym.Env):
    def __init__(self, goal_length, train_iter, 
                 p_eps=0.05, 
                 teacher_reward=1,
                 student_reward=1,
                 student_qe_dist=None,
                 **student_args):
        super().__init__()

        self.student = None
        self.goal_length = goal_length
        self.train_iter = train_iter
        self.p_eps = p_eps
        self.teacher_reward = teacher_reward
        self.student_reward = student_reward
        self.student_qe_dist = student_qe_dist

        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(goal_length), 
            gym.spaces.Box(low=0, high=1, shape=(1,))))
        self.N = goal_length

        self.action_space = gym.spaces.Discrete(3)
        self.student_args = student_args
    
    def step(self, action):
        d_length = action - 1
        self.N = np.clip(self.N + d_length, 1, self.goal_length)
        prob = np.exp(self._get_score(self.N))

        reward = 0
        is_done = False

        if self.N == self.goal_length and (1 - prob) < self.p_eps:
            reward = self.teacher_reward
            is_done = True
        
        prob = self._get_score(self.N)
        return (self.N, prob), reward, is_done, {}
    
    def reset(self):
        if self.student_qe_dist == None:
            self.student = Student(**self.student_args)
        else:
            qe_val = self.student_qe_dist()
            q_e = defaultdict(lambda: qe_val)
            self.student = Student(q_e=q_e, **self.student_args)

        student_score = self._get_score(self.goal_length)
        self.N  = self.goal_length
        return (self.goal_length, student_score)

    def _get_score(self, length):
        self.student.learn(BinaryEnv(length, reward=self.student_reward), max_iters=self.train_iter)
        return self.student.score(length)


class Agent:
    def __init__(self) -> None:
        self.iter = 0

    def next_action(state):
        raise NotImplementedError('next_action not implemented in Agent')

    def update(old_state, action, reward, next_state, is_done):
        raise NotImplementedError('update not implemented in Agent')
    
    def learn(self, env, 
             max_iters=1000, 
             use_tqdm=False, 
             post_hook=None, done_hook=None):
        state = env.reset()
        self.iter = 0

        iterator = range(max_iters)
        if use_tqdm:
            iterator = tqdm(iterator)

        for _ in iterator:
            action = self.next_action(state)
            next_state, reward, is_done, _ = env.step(action)
            self.update(state, action, reward, next_state, is_done)

            if post_hook != None:
                post_hook(self)

            if is_done:
                if done_hook != None:
                    done_hook(self)
                state = env.reset()
            else:
                state = next_state
            
            self.iter += 1


class Student(Agent):
    def __init__(self, lr=0.1, gamma=1, q_e=None) -> None:
        super().__init__()
        self.lr = lr
        self.gamma = gamma

        # only track Q-values for action = 1, maps state --> value
        self.q_e = defaultdict(int) if q_e == None else q_e
        self.q_r = defaultdict(int)
    
    # softmax policy
    def policy(self, state) -> np.ndarray:
        q = self.q_e[state] + self.q_r[state]
        prob = sigmoid(q)
        return np.array([1 - prob, prob])

    def next_action(self, state) -> int:
        _, prob = self.policy(state)
        return np.random.binomial(n=1, p=prob)
    
    def update(self, old_state, _, reward, next_state, is_done):
        _, prob = self.policy(next_state)
        if is_done:
            exp_q = 0
        else:
            exp_q = prob * self.q_r[next_state]
        self.q_r[old_state] += self.lr * (reward + self.gamma * exp_q - self.q_r[old_state])

    def score(self, goal_state) -> float:
        qs = [self.q_e[s] + self.q_r[s] for s in range(goal_state)]
        log_prob = np.sum([-np.log(1 + np.exp(-q)) for q in qs])
        return log_prob


class Teacher(Agent):
    def __init__(self, lr=0.1, gamma=1, bins=20, anneal_sched=None) -> None:
        super().__init__()

        self.lr = lr
        self.gamma = gamma
        # self.q = defaultdict(int)
        self.q = {((n, l), a):0 for n in np.arange(1, 11) for l in np.arange(0, bins + 1) for a in [0, 1, 2]}
        self.bins = bins
        self.anneal_sched = anneal_sched
    
    def _to_bin(self, state, logit_min=-2, logit_max=2, eps=1e-8):
        log_p = state[1]
        logit = log_p - np.log(1 - np.exp(log_p) + eps)

        norm = (logit - logit_min) / (logit_max - logit_min)
        bin_p = np.clip(np.round(norm * self.bins), 0, self.bins)
        
        return (state[0], bin_p)
    
    # softmax policy
    def policy(self, state_bin) -> np.ndarray:
        if self.anneal_sched == None:
            beta = 1
        else:
            beta = self.anneal_sched(self.iter)

        qs = np.array([self.q[(state_bin, a)] for a in [0, 1, 2]])
        probs = np.exp(beta * qs) / np.sum(np.exp(beta * qs))
        return probs
    
    def next_action(self, state, is_binned=False):
        state = self._to_bin(state) if not is_binned else state
        probs = self.policy(state)
        try:
            return np.random.choice([0, 1, 2], p=probs)
        except ValueError as e:
            print('probs', probs)
            print('state', state)
            qs = np.array([self.q[(state, a)] for a in [0, 1, 2]])
            print('qs:', qs)
            raise e


    def update(self, old_state, action, reward, next_state, is_done):
        old_state = self._to_bin(old_state)
        next_state = self._to_bin(next_state)

        if is_done:
            exp_q = 0
        else:
            probs = self.policy(next_state)
            qs = np.array([self.q[next_state, a] for a in [0, 1, 2]])
            exp_q = np.sum(probs * qs)

        self.q[old_state, action] += self.lr * (reward + self.gamma * exp_q - self.q[old_state, action])
    

class NaiveTest:
    def __init__(self, goal_length):
        self.goal_length = goal_length
    
    def run(self, student, T, max_iters=1000, student_reward=1):
        self.iter = 0

        for _ in range(max_iters):
            student.learn(BinaryEnv(self.goal_length, reward=student_reward), max_iters=T)
            self.iter += 1

            final_score = student.score(self.goal_length)
            if np.isclose(final_score, 0, atol=1e-1):
                break
        
        return self.iter


class TeacherHeuristicTest:
    def __init__(self, goal_length, lr=0.1) -> None:
        self.q = np.zeros(goal_length)
        self.lr = lr

    # softmax policy
    def policy(self) -> np.ndarray:
        probs = np.exp(self.q) / np.sum(np.exp(self.q))
        return probs
    
    def next_action(self):
        probs = self.policy()
        return np.random.choice(len(probs), p=probs)

    def update(self, state, reward):
        self.q[state] = self.lr * reward + (1 - self.lr) * self.q[state]
    
    def run(self, student, T, max_iters=1000, student_reward=1, scale=100):
        self.iter = 0

        for _ in range(max_iters):
            N = self.next_action() + 1
            scores = []
            student.learn(BinaryEnv(N, reward=student_reward), max_iters=T, 
                          post_hook=lambda s: scores.append(s.score(N)))
            
            slope, _, _, _, _ = linregress(np.arange(len(scores)), scores)
            slope *= scale
            self.update(N - 1, np.abs(slope))
            self.iter += 1

            final_score = student.score(len(self.q))
            if np.isclose(final_score, 0, atol=1e-1):
                break
        
        return self.iter


class TeacherAgentTest:
    def __init__(self, teacher_agent, goal_length):
        self.teacher = teacher_agent
        self.goal_length = goal_length
    
    def run(self, student, T, max_iters=1000, student_reward=1):
        self.iter = 0
        N = self.goal_length

        for _ in range(max_iters):
            log_p = student.score(N)
            a = teacher.next_action((N, log_p)) - 1
            N = np.clip(N + a, 1, self.goal_length)
            print(N)
            student.learn(BinaryEnv(N, reward=student_reward), max_iters=T)
            self.iter += 1

            final_score = student.score(self.goal_length)
            if np.isclose(final_score, 0, atol=1e-1):
                break
        
        return self.iter


'''

# <codecell>
# teacher = TeacherHeuristic(10)
# student = Student()
# iters = teacher.learn(student, 20, max_iters=1000, student_reward=10)

# teacher = NaiveTest(10)
test = TeacherHeuristicTest(10)
student = Student()
iters = test.run(student, 20, max_iters=10000, student_reward=10)

print(iters)

# %%

student = Student()
scores = []
student.learn(BinaryEnv(10, reward=10), max_iters=1100 * 20, post_hook=lambda s: scores.append(s.score(10)))
# %%
import matplotlib.pyplot as plt
plt.plot(scores)
# %%
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
# %%
# TODO: debug possible issues and make comparison plots
test = TeacherAgentTest(teacher, 10)
student = Student()
iters = test.run(student, T, max_iters=10000, student_reward=student_reward)
print(iters)
'''