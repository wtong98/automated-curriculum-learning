"""
Simple binary environments for experimenting with teacher-student interactions

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from collections import defaultdict
import gym
import numpy as np
import matplotlib.pyplot as plt

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
        
        log_prob = self._get_score(self.N)
        return (self.N, log_prob), reward, is_done, {}
    
    def reset(self):
        if self.student_qe_dist == None:
            self.student = Student(**self.student_args)
        else:
            qe_val = self.student_qe_dist()
            q_e = defaultdict(lambda: qe_val)
            self.student = Student(q_e=q_e, **self.student_args)

        student_score = self._get_score(self.goal_length)
        self.N  = 1
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
    
    def learn(self, env, is_eval=False,
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
            if not is_eval:
                self.update(state, action, reward, next_state, is_done)

            if post_hook != None:
                post_hook(self)

            if is_done:
                if done_hook != None:
                    done_hook(self, reward)
                state = env.reset()
            else:
                state = next_state
            
            self.iter += 1


# TODO: include score as observed number of successes
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
        self.q = defaultdict(int)
        # self.q = {((n, l), a):0 for n in np.arange(1, 11) for l in np.arange(0, bins + 1) for a in [0, 1, 2]}
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
    
    def plot_q(self, N):
        ns = np.arange(N) + 1
        ls = np.arange(0, self.bins + 1)

        ll, nn = np.meshgrid(ls, ns)
        actions = []

        for l, n in zip(ll.ravel(), nn.ravel()):
            a = self.next_action((n, l), is_binned=True)
            actions.append(a)

        z = np.array(actions).reshape(ll.shape) - 1
        # plt.contourf(ll, nn, z)
        plt.imshow(z)
        plt.colorbar()
    
    def plot_q_ent(self, N):
        ns = np.arange(N) + 1
        ls = np.arange(0, self.bins + 1)
        entropy = []
        ll, nn = np.meshgrid(ls, ns)

        for l, n in zip(ll.ravel(), nn.ravel()):
            probs = self.policy((n, l))
            entropy.append(-np.sum(probs * np.log(probs)))

        z = np.array(entropy).reshape(ll.shape)
        # plt.contourf(ll, nn, z)
        plt.imshow(z)
        plt.colorbar()

# <codecell>
# student = Student()
# student.learn(BinaryEnv(5), is_eval=False)
# %%
