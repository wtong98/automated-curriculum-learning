"""
Simple binary environments for experimenting with teacher-student interactions

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
from collections import defaultdict, namedtuple
import gym
import numpy as np
import matplotlib.pyplot as plt

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
        self.N = 1

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
        
        # log_prob = self._get_score(self.N)
        log_prob = np.log(prob)
        return (self.N, log_prob), reward, is_done, {}
    
    def reset(self):
        if self.student_qe_dist == None:
            self.student = Student(**self.student_args)
        else:
            if type(self.student_qe_dist) == int or type(self.student_qe_dist) == float:
                qe_val = np.random.normal(0, self.student_qe_dist)
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
    def __init__(self, lr=0.05, gamma=1, q_e=None) -> None:
        super().__init__()
        self.lr = lr
        self.gamma = gamma

        # only track Q-values for action = 1, maps state --> value
        self.q_e = defaultdict(int) if type(q_e) == type(None) else q_e
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
            if type(self.anneal_sched) == int or type(self.anneal_sched) == float:
                beta = self.iter * self.anneal_sched / 100000   # TODO: hardcoded max_iters
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


class TeacherPomcpAgent(Agent):
    def __init__(self, goal_length, T, bins=10, p_eps=0.05, student_qe=0, student_lr=0.01, student_reward=10, n_particles=500, gamma=0.9, eps=1e-2, explore_factor=1) -> None:
        super().__init__()
        self.goal_length = goal_length
        self.T = T
        self.bins = bins
        self.p_eps = p_eps
        self.student_qe = student_qe
        self.student_lr = student_lr
        self.student_reward = student_reward

        self.n_particles = n_particles
        self.gamma = gamma
        self.eps = eps
        self.explore_factor = explore_factor

        self.actions = [0, 1, 2]
        self.history = ()
        self.tree = {}

        self.qrs_means = []
        self.qrs_stds = []
        self.qrs_true = []
        self.num_particles = []
    
    
    def reset(self):
        self.history = ()
        self.tree = {}

        self.qrs_means = []
        self.qrs_stds = []
        self.qrs_true = []
        self.num_particles = []
    

    def next_action(self, prev_action=None, obs=None):
        if obs != None and prev_action != None:
            print('Observed:', obs)
            self.history += (prev_action, obs,)

            qrs = np.array([qr for _, qr in self.tree[self.history]['b']])
            qrs_mean = np.mean(qrs, axis=0)
            qrs_std = np.std(qrs, axis=0)

            self.num_particles.append(len(qrs))
            self.qrs_means.append(qrs_mean)
            self.qrs_stds.append(qrs_std)
            # self.qrs_true.append([env.student.q_r[i] for i in range(2)])

        return self._search()

    def _sample_transition(self, state, action):
        n, qr = state
        new_n = np.clip(n + action - 1, 1, self.goal_length)

        qs = [self.student_qe[s] + qr[s] for s in range(self.goal_length)]
        log_trans_probs = [-np.log(1 + np.exp(-q)) for q in qs]
        success_prob = np.exp(np.cumsum(log_trans_probs))
        num_contacts = success_prob * self.T

        diff_qs = np.zeros(self.goal_length)
        upd_idx = new_n - 1
        diff_qs[:upd_idx] = (self.student_lr * num_contacts[:-1] \
            * (np.exp(log_trans_probs[1:]) * (qs[1:] - self.student_qe[1:]) + self.student_qe[:-1] - qs[:-1]))[:upd_idx]
        diff_qs[upd_idx] = self.student_lr * num_contacts[upd_idx] * (self.student_reward + self.student_qe[upd_idx] - qs[upd_idx])
        new_qr = qr + diff_qs
        new_qs = qs + diff_qs

        # print('---')
        # print('ACTION', action)
        # print('ORIG_QR', qr)
        # print('N_CONTA', num_contacts)
        # print('DIFF_QS', diff_qs)
        # print('NEW_QR', new_qr)
        # print('LR:', self.student_lr)

        log_prob = np.sum([-np.log(1 + np.exp(-q)) for q in new_qs[:new_n]])
        obs = self._to_bin(log_prob)

        reward = 0
        if new_n == self.goal_length and 1 - np.exp(log_prob) < self.p_eps:
            reward = self.student_reward

        return (new_n, new_qr), obs, reward

    # TODO: copied from teacher
    def _to_bin(self, log_p, logit_min=-2, logit_max=2, eps=1e-8):
        logit = log_p - np.log(1 - np.exp(log_p) + eps)

        norm = (logit - logit_min) / (logit_max - logit_min)
        bin_p = np.clip(np.round(norm * self.bins), 0, self.bins)
        
        return bin_p

    def _sample_prior(self):
        qr = np.random.normal(scale=0.5, size=self.goal_length)
        return (1, qr) 

    def _sample_rollout_policy(self, history):
        return np.random.choice(self.actions)   # TODO: use something better?
    
    def _init_node(self):
        return {'v': 0, 'n': 0, 'b': []}
    
    def _search(self):
        for _ in range(self.n_particles):
            if len(self.history) == 0:
                state = self._sample_prior()
            else:
                if np.random.random() < 0.3:   # particle reinvigoration prob
                    qrs = [qr for _, qr in self.tree[self.history]['b']]
                    qrs_mean = np.mean(qrs, axis=0)
                    qrs_cov = np.cov(qrs, rowvar=False)
                    np.fill_diagonal(qrs_cov, np.clip(np.diag(qrs_cov), 0.2, np.inf))  # boost variance

                    # print('HIST LEN', len(self.tree[self.history]['b']))
                    # print('QRS_COV', qrs_cov)
                    # print('QRS_CORR', np.corrcoef(qrs, rowvar=False))
                    new_qrs = np.random.multivariate_normal(qrs_mean, qrs_cov)
                    state = (self.tree[self.history]['b'][0][0], new_qrs)
                    # print('Invigorated', state)
                else:
                    state_idx = np.random.choice(len(self.tree[self.history]['b']))
                    state = self.tree[self.history]['b'][state_idx]
        
            self._simulate(state, self.history, 0)
        
        vals = [self.tree[self.history + (a,)]['v'] for a in self.actions]
        print('VALS', vals)
        return np.argmax(vals)
    
    def _simulate(self, state, history, depth):
        if self.gamma ** depth < self.eps:
            return 0
        
        if history not in self.tree:
            self.tree[history] = self._init_node()
            for a in self.actions:
                proposal = history + (a,)
                self.tree[proposal] = self._init_node()
            return self._rollout(state, history, depth)
        
        vals = []
        for a in self.actions:
            curr_node = self.tree[history]
            next_node = self.tree[history + (a,)]
            if curr_node['n'] > 0 and next_node['n'] > 0:
                explore = self.explore_factor * np.sqrt(np.log(curr_node['n']) / next_node['n'])
            else:
                explore = 999  # arbitrarily high

            vals.append(next_node['v'] + explore)
        
        a = np.argmax(vals)
        next_state, obs, reward = self._sample_transition(state, a)
        total_reward = reward + self.gamma * self._simulate(next_state, history + (a, obs), depth + 1)

        if depth > 0:   # NOTE: avoid re-adding encountered state?
            self.tree[history]['b'].append(state)
            self.tree[history]['n'] += 1

        next_node = self.tree[history + (a,)]
        next_node['n'] += 1
        next_node['v'] += (total_reward - next_node['v']) / next_node['n']
        return total_reward
    
    def _rollout(self, state, history, depth):
        if self.gamma ** depth < self.eps:
            return 0
        
        a = self._sample_rollout_policy(history)
        next_state, obs, reward = self._sample_transition(state, a)
        return reward + self.gamma * self._rollout(next_state, history + (a, obs), depth + 1)

    def learn(self, *args, **kwargs):
        raise NotImplementedError('TeacherPomcpAgent does not implement method `learn`')


# '''
# <codecell>
N = 3
T = 10
student_lr = 0.005
p_eps = 0.1
L = 20
gamma = 0.9
es = np.zeros(N+1) - 1

qrs_true = []

agent = TeacherPomcpAgent(goal_length=N, T=T, bins=L, p_eps=p_eps, student_qe=es, student_lr=student_lr, gamma=gamma)
env = CurriculumEnv(goal_length=N, train_iter=T, p_eps=p_eps, teacher_reward=10, student_reward=10, lr=student_lr, q_e=es)

prev_obs = env.reset()
prev_a = None
is_done = False


while not is_done:
    a = agent.next_action(prev_a, prev_obs)
    print('Took a:', a)

    state, reward, is_done, _ = env.step(a)
    obs = agent._to_bin(state[1])
    print('At n:', env.N)

    qrs_true.append((env.student.q_r[0], env.student.q_r[1]))
    prev_a = a
    prev_obs = obs

qrs_true = qrs_true[1:]
print('Great success!')

# <codecell>
### PLOT RUN DIAGNOSTICS
fig, axs = plt.subplots(2, 1, figsize=(6, 6))

steps = np.arange(len(agent.num_particles))

axs[0].errorbar(steps, [q[0] for q in agent.qrs_means], yerr=[2 * s[0] for s in agent.qrs_stds], label='Pred qr[0]')
axs[0].errorbar(steps, [q[1] for q in agent.qrs_means], yerr=[2 * s[1] for s in agent.qrs_stds], label='Pred qr[1]', color='C0')
axs[0].plot(steps, [q[0] for q in qrs_true], label='True qr[0]')
axs[0].plot(steps, [q[1] for q in qrs_true], label='True qr[1]', color='C1')
axs[0].legend()
axs[0].set_xlabel('Step')
axs[0].set_ylabel('q')

axs[1].bar(steps, agent.num_particles)
axs[1].set_xlabel('Step')
axs[1].set_ylabel('# particles')

fig.suptitle('State estimates of POMCP agent')
fig.tight_layout()

plt.savefig('fig/pomcp_state_estimate.png')


# %%
# <codecell>
### INVESTIGATE CLOSENESS OF MODEL TO REALITY
N = 5
T = 10
student_lr = 0.005 
p_eps = 0.1
L = 20
es = np.zeros(N+1) - 1   # TODO: raw conversion seems to cause problems here

agent = TeacherPomcpAgent(goal_length=N, T=T, bins=10, p_eps=p_eps, student_qe=es, student_lr=student_lr)
env = CurriculumEnv(goal_length=N, train_iter=T, p_eps=p_eps, teacher_reward=10, student_reward=10, lr=student_lr, q_e=es)

iters = 20

all_pred_qrs = []
all_pred_obs = []
all_qrs = []
all_obs = []

env.reset()

for _ in range(iters):
    action = np.random.choice(3)

    q_r = env.student.q_r
    n = env.N

    q_r = [q_r[i] for i in range(N)]
    (new_n, new_qr), pred_obs, pred_reward = agent._sample_transition((n, q_r), action)

    state, reward, _, _ = env.step(action)

    # print('-T-R-U-E-')
    # print('QR', env.student.q_r)

    assert state[0] == new_n
    assert reward == pred_reward

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

# %%
'''
