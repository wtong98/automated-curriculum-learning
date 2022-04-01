"""
Simple binary environments for experimenting with teacher-student interactions

author: William Tong (wtong@g.harvard.edu)
"""

# <codecell>
import copy

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

        student_score = self._get_score(self.goal_length, train=False)
        self.N  = 1
        return (self.goal_length, student_score)

    def _get_score(self, length, train=True):
        if train:
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
        if is_done:
            exp_q = 0
        else:
            _, prob = self.policy(next_state)
            exp_q = prob * (self.q_e[next_state] + self.q_r[next_state])  # TODO: double-check
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
    def __init__(self, goal_length, T, bins=10, p_eps=0.05, student_qe=0, student_lr=0.01, student_reward=10, n_particles=500, gamma=0.9, eps=1e-2, explore_factor=1, q_reinv_var=0.3) -> None:
        super().__init__()
        self.goal_length = goal_length
        self.T = T
        self.bins = bins
        self.p_eps = p_eps
        self.student_qe = student_qe
        self.student_lr = student_lr
        self.student_reward = student_reward
        self.q_reinv_var = q_reinv_var

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
        self.replicas = []
    
    
    def reset(self):
        self.history = ()
        self.tree = {}

        self.qrs_means = []
        self.qrs_stds = []
        self.qrs_true = []
        self.num_particles = []
    

    def next_action(self, prev_action=None, obs=None, with_replicas=0):
        if obs != None and prev_action != None:
            print(f'Observed: {obs}')
            self.history += (prev_action, obs,)

            qrs = np.array([qr for _, qr, _ in self.tree[self.history]['b']])
            qrs_mean = np.mean(qrs, axis=0)
            qrs_std = np.std(qrs, axis=0)

            self.num_particles.append(len(qrs))
            self.qrs_means.append(qrs_mean)
            self.qrs_stds.append(qrs_std)
            # self.qrs_true.append([env.student.q_r[i] for i in range(2)])
            print('N_particles:', len(qrs))

        all_a = []
        for _ in range(with_replicas):
            tmp_tree = copy.deepcopy(self.tree)
            a = self._search()
            all_a.append(a)
            self.tree = tmp_tree
        
        a = self._search()
        all_a.append(a)
        self.replicas.append(all_a)
        return a

    def _sample_transition(self, state, action):
        n, qr, success_states = state
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
        is_done = False
        # if new_n == self.goal_length and 1 - np.exp(log_prob) < self.p_eps:
        #     reward = self.student_reward
        #     is_done = True
        if 1 - np.exp(log_prob) < self.p_eps:
            if new_n == self.goal_length:
                reward = self.student_reward
                is_done = True
            elif len(success_states) == 0 or new_n > np.max(success_states):
                    reward = 1   # intermediate reward boost
                    success_states = success_states + (new_n,)
                    # print('REWARDED FOR', new_n)
                    # print('SUCESS_STATES', success_states)

        return (new_n, new_qr, success_states), obs, reward, is_done

    # TODO: copied from teacher
    def _to_bin(self, log_p, logit_min=-2, logit_max=2, eps=1e-8):
        logit = log_p - np.log(1 - np.exp(log_p) + eps)

        norm = (logit - logit_min) / (logit_max - logit_min)
        bin_p = np.clip(np.round(norm * self.bins), 0, self.bins)
        
        return bin_p

    def _sample_prior(self):
        qr = np.random.normal(scale=0.5, size=self.goal_length)
        return (1, qr, ()) 

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
                    qrs = [state[1] for state in self.tree[self.history]['b']]
                    qrs_mean = np.mean(qrs, axis=0)
                    qrs_cov = np.cov(qrs, rowvar=False)
                    np.fill_diagonal(qrs_cov, np.clip(np.diag(qrs_cov), self.q_reinv_var, np.inf))  # boost variance
                    new_qrs = np.random.multivariate_normal(qrs_mean, qrs_cov)
                    state_idx = np.random.choice(len(self.tree[self.history]['b']))

                    # TODO: confirm that attributes are evolving correctly <-- STOPPED HERE
                    samp_state = self.tree[self.history]['b'][state_idx]   # randomly select fixed attributes
                    state = (samp_state[0], new_qrs, samp_state[-1])
                else:
                    state_idx = np.random.choice(len(self.tree[self.history]['b']))
                    state = self.tree[self.history]['b'][state_idx]
        
            self._simulate(state, self.history, 0)
        
        vals = [self.tree[self.history + (a,)]['v'] for a in self.actions]
        print('VALS', vals)
        return np.argmax(vals)
    
    def _simulate(self, state, history, depth):
        reward_stack = []
        node_stack = []
        n_visited_stack = []

        while self.gamma ** depth > self.eps:
            if history not in self.tree:
                self.tree[history] = self._init_node()
                for a in self.actions:
                    proposal = history + (a,)
                    self.tree[proposal] = self._init_node()
                pred_reward = self._rollout(state, history, depth)
                reward_stack.append(pred_reward)
                break

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
            next_state, obs, reward, is_done = self._sample_transition(state, a)
            reward_stack.append(reward)
            if is_done:
                break

            if depth > 0:   # NOTE: avoid re-adding encountered state
                self.tree[history]['b'].append(state)
                self.tree[history]['n'] += 1

            next_node = self.tree[history + (a,)]
            next_node['n'] += 1
            node_stack.append(next_node)
            n_visited_stack.append(next_node['n'])
            # next_node['v'] += (total_reward - next_node['v']) / next_node['n']

            history += (a, obs)
            state = next_state
            depth += 1
            # total_reward = reward + self.gamma * self._simulate(next_state, history + (a, obs), depth + 1)
        
        for i, (node, n_visited) in enumerate(zip(node_stack, n_visited_stack)):
            total_reward = np.sum([r * self.gamma ** iters for iters, r in enumerate(reward_stack[i:])])
            node['v'] += (total_reward - node['v']) / n_visited


        # if self.gamma ** depth < self.eps:
        #     return 0
        
        # if history not in self.tree:
        #     self.tree[history] = self._init_node()
        #     for a in self.actions:
        #         proposal = history + (a,)
        #         self.tree[proposal] = self._init_node()
        #     return self._rollout(state, history, depth)
        
        # vals = []
        # for a in self.actions:
        #     curr_node = self.tree[history]
        #     next_node = self.tree[history + (a,)]
        #     if curr_node['n'] > 0 and next_node['n'] > 0:
        #         explore = self.explore_factor * np.sqrt(np.log(curr_node['n']) / next_node['n'])
        #     else:
        #         explore = 999  # arbitrarily high

        #     vals.append(next_node['v'] + explore)
        
        # a = np.argmax(vals)
        # next_state, obs, reward = self._sample_transition(state, a)
        # total_reward = reward + self.gamma * self._simulate(next_state, history + (a, obs), depth + 1)

        # if depth > 0:   # NOTE: avoid re-adding encountered state?
        #     self.tree[history]['b'].append(state)
        #     self.tree[history]['n'] += 1

        # next_node = self.tree[history + (a,)]
        # next_node['n'] += 1
        # next_node['v'] += (total_reward - next_node['v']) / next_node['n']
        # return total_reward
    
    def _rollout(self, state, history, depth):
        g = 1
        total_reward = 0

        while self.gamma ** depth > self.eps:
            a = self._sample_rollout_policy(history)
            state, obs, reward, is_done = self._sample_transition(state, a)

            history += (a, obs)
            total_reward += g * reward
            g *= self.gamma
            depth += 1

            if is_done:
                break
        
        return total_reward

        # depth = init_depth
        # if self.gamma ** depth < self.eps:
        #     return 0
        
        # a = self._sample_rollout_policy(history)
        # next_state, obs, reward = self._sample_transition(state, a)
        # return reward + self.gamma * self._rollout(next_state, history + (a, obs), depth + 1)

    def learn(self, *args, **kwargs):
        raise NotImplementedError('TeacherPomcpAgent does not implement method `learn`')


# '''
# <codecell>
N = 10
T = 20
student_lr = 0.005
p_eps = 0.1
L = 5
gamma = 0.95
es = np.zeros(N)

qrs_true = []

agent = TeacherPomcpAgent(goal_length=N, T=T, bins=L, p_eps=p_eps, student_qe=es, student_lr=student_lr, gamma=gamma, n_particles=1000)
env = CurriculumEnv(goal_length=N, train_iter=T, p_eps=p_eps, teacher_reward=10, student_reward=10, lr=student_lr, q_e=es)

prev_obs = env.reset()
prev_a = None
is_done = False


while not is_done:
    a = agent.next_action(prev_a, prev_obs, with_replicas=0)

    state, reward, is_done, _ = env.step(a)
    obs = agent._to_bin(state[1])
    print(f'Took a: {a}   At n: {env.N}')

    qrs_true.append([env.student.q_r[i] for i in range(N)])
    prev_a = a
    prev_obs = obs

qrs_true = qrs_true[1:]
print('Great success!')

# <codecell>
### PLOT RUN DIAGNOSTICS
fig, axs = plt.subplots(2, 1, figsize=(6, 6))

steps = np.arange(len(agent.num_particles))

for i in range(N):
    axs[0].errorbar(steps, [q[i] for q in agent.qrs_means], yerr=[2 * s[i] for s in agent.qrs_stds], color=f'C{i}', alpha=0.5, fmt='o', markersize=0)
    axs[0].plot(steps, [q[i] for q in qrs_true], label=f'qr[{i}]', color=f'C{i}')

# axs[0].errorbar(steps, [q[0] for q in agent.qrs_means], yerr=[2 * s[0] for s in agent.qrs_stds], label='Pred qr[0]')
# axs[0].errorbar(steps, [q[1] for q in agent.qrs_means], yerr=[2 * s[1] for s in agent.qrs_stds], label='Pred qr[1]', color='C0')
# axs[0].plot(steps, [q[0] for q in qrs_true], label='True qr[0]')
# axs[0].plot(steps, [q[1] for q in qrs_true], label='True qr[1]', color='C1')
axs[0].legend()
axs[0].set_xlabel('Step')
axs[0].set_ylabel('q')

axs[1].bar(steps, agent.num_particles)
axs[1].set_xlabel('Step')
axs[1].set_ylabel('# particles')

fig.suptitle('State estimates of POMCP agent')
fig.tight_layout()

plt.savefig('fig/pomcp_state_estimate.png')

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
T = 20
student_lr = 0.005 
p_eps = 0.1
L = 5
es = np.zeros(N)

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
    ss = ()

    q_r = [q_r[i] for i in range(N)]
    (new_n, new_qr, ss), pred_obs, pred_reward, _ = agent._sample_transition((n, q_r, ss), action)

    state, reward, _, _ = env.step(action)

    # print('-T-R-U-E-')
    # print('QR', env.student.q_r)

    assert state[0] == new_n
    # assert reward == pred_reward

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
