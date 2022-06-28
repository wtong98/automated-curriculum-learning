"""
Simple binary environments for experimenting with teacher-student interactions

author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>
import copy

from collections import defaultdict
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
    def __init__(self, goal_length=10, train_iter=50, 
                 p_eps=0.05, 
                 teacher_reward=1,
                 student_reward=1,
                 student_qe_dist=None,
                 student_params=None,
                 anarchy_mode=False):
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
        self.student_params = student_params if student_params != None else {}
        self.anarchy_mode = anarchy_mode
    
    def step(self, action):
        if self.anarchy_mode:
            self.N = action
        else:
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
        self.student = Student(q_e=self.student_qe_dist, **self.student_params)
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
        if type(q_e) == int or type(q_e) == float:
            self.q_e = defaultdict(lambda: q_e)
        elif type(q_e) != type(None):
            self.q_e = q_e
        else:
            self.q_e = defaultdict(int)
        self.q_r = defaultdict(int)
    
    # softmax policy
    def policy(self, state) -> np.ndarray:
        q = self.q_e[state] + self.q_r[state]
        prob = sigmoid(q)
        return np.array([1 - prob, prob])

    def next_action(self, state) -> int:
        _, prob = self.policy(state)
        a = np.random.binomial(n=1, p=prob)
        return a 
    
    def update(self, old_state, _, reward, next_state, is_done):
        if is_done:
            exp_q = 0
        else:
            _, prob = self.policy(next_state)
            # exp_q = prob * (self.q_e[next_state] + self.q_r[next_state])
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
            qs = np.array([self.q[(state, a)] for a in [0, 1, 2]])
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
    def __init__(self, goal_length, T, bins=10, p_eps=0.05, lookahead_cap=None,
                       student_qe=0, student_lr=0.01, student_reward=10, 
                       n_particles=500, gamma=0.9, eps=1e-2, 
                       explore_factor=1, q_reinv_scale=1.5, q_reinv_prob=0.25) -> None:
        super().__init__()
        self.goal_length = goal_length
        self.T = T
        self.bins = bins
        self.p_eps = p_eps
        self.lookahead_cap = lookahead_cap
        self.student_qe = student_qe
        self.student_lr = student_lr
        self.student_reward = student_reward
        self.q_reinv_scale = q_reinv_scale
        self.q_reinv_prob = q_reinv_prob

        self.n_particles = n_particles
        self.gamma = gamma
        self.eps = eps
        self.explore_factor = explore_factor

        self.actions = [0, 1, 2]
        self.history = ()
        self.tree = {}

        self.curr_n = 1
        self.qrs_means = []
        self.qrs_stds = []
        self.num_particles = []
        self.replicas = []
    
    
    def reset(self):
        self.history = ()
        self.tree = {}

        self.qrs_means = []
        self.qrs_stds = []
        self.num_particles = []
        self.curr_n = 1
    

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

        self.curr_n = np.clip(self.curr_n + a - 1, 1, self.goal_length)  # NOTE: assuming agent follows the proposed action
        return a

    def _sample_transition(self, state, action):
        n, qr, success_states = state
        lookahead_len = self.goal_length
        if self.lookahead_cap != None:
            lookahead_len = min(self.curr_n + self.lookahead_cap, self.goal_length)
        
        new_n = np.clip(n + action - 1, 1, lookahead_len)

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

        log_prob = np.sum([-np.log(1 + np.exp(-q)) for q in new_qs[:new_n]])
        obs = self._to_bin(log_prob)

        reward = 0
        is_done = False

        if 1 - np.exp(log_prob) < self.p_eps:
            if new_n == self.goal_length:
                reward = self.student_reward
                is_done = True
            elif new_n == lookahead_len:
                reward = 1   # intermediate reward boost at lookahead cap
                is_done = True
            elif len(success_states) == 0 or new_n > np.max(success_states):
                reward = 0   # intermediate reward boost for successful finish
                success_states = success_states + (new_n,)

        return (new_n, new_qr, success_states), obs, reward, is_done

    # TODO: copied from teacher
    def _to_bin(self, log_p, logit_min=-2, logit_max=2, eps=1e-8):
        logit = log_p - np.log(1 - np.exp(log_p) + eps)

        norm = (logit - logit_min) / (logit_max - logit_min)
        bin_p = np.clip(np.round(norm * self.bins), 0, self.bins)
        
        return bin_p

    def _sample_prior(self):
        qr = np.random.normal(scale=0.1, size=self.goal_length)
        return (1, qr, ()) 

    def _sample_rollout_policy(self, history):
        return np.random.choice(self.actions)   # TODO: use something better?
    
    def _init_node(self):
        return {'v': 0, 'n': 0, 'b': []}
    
    def _search(self):
        if len(self.history) == 0:
            for _ in range(self.n_particles):
                state = self._sample_prior()
                self._simulate(state, self.history, 0)
        else:
            qrs = [state[1] for state in self.tree[self.history]['b']]
            qrs_mean = np.mean(qrs, axis=0)
            qrs_cov = self.q_reinv_scale * np.cov(qrs, rowvar=False)

            for _ in range(self.n_particles):
                state_idx = np.random.choice(len(self.tree[self.history]['b']))
                
                if np.random.random() < self.q_reinv_prob:
                    new_qrs = np.random.multivariate_normal(qrs_mean, qrs_cov)
                    samp_state = self.tree[self.history]['b'][state_idx]   # randomly select fixed attributes
                    state = (samp_state[0], new_qrs, samp_state[2])
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

            history += (a, obs)
            state = next_state
            depth += 1
        
        for i, (node, n_visited) in enumerate(zip(node_stack, n_visited_stack)):
            total_reward = np.sum([r * self.gamma ** iters for iters, r in enumerate(reward_stack[i:])])
            node['v'] += (total_reward - node['v']) / n_visited


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


    def learn(self, *args, **kwargs):
        raise NotImplementedError('TeacherPomcpAgent does not implement method `learn`')
    

class TeacherPerfectKnowledge(Agent):
    def __init__(self, goal_length, T, p_eps=0.05, 
                       student_qe=0, student_lr=0.01, student_reward=10, 
                       n_iters=500, gamma=0.9, eps=1e-2, explore_factor=1) -> None:
        super().__init__()
        self.goal_length = goal_length
        self.T = T
        self.p_eps = p_eps
        self.student_qe = student_qe
        self.student_lr = student_lr
        self.student_reward = student_reward

        self.n_iters = n_iters
        self.gamma = gamma
        self.eps = eps
        self.explore_factor = explore_factor

        self.actions = np.arange(goal_length) + 1
        self.history = ()
        self.tree = {}
    
    def reset(self):
        self.history = ()
        self.tree = {}

        self.qrs_means = []
        self.qrs_stds = []
        self.num_particles = []
    

    def next_action(self, prev_action=None, qr=None):
        if prev_action != None and type(qr) != type(None):
            qr = self._round(qr)
            self.history += (prev_action, tuple(qr))
            if self.history not in self.tree:
                print('warn: rerooting tree')
                self.tree = {}
                self.history = self.history[-2:]
        else:
            self.history += (1, tuple(np.zeros(self.goal_length)))

        a = self._search()
        return a
    
    def _round(self, val):
        return np.round(val, decimals=1)
    
    # def _update_qr(self, n, qr, fail_idx):
    #     if fail_idx == n + 1:
    #         payoff = self.student_reward
    #     else:
    #         payoff = 0
        
    #     rpe = np.append(qr[1:fail_idx+1], payoff) - qr[:fail_idx+1]
    #     qr[:fail_idx+1] += self.student_lr * (rpe - qr[:fail_idx+1])
    #     return qr
    
    # def _sample_run(self, n, qr):
    #     qs = self.student_qe + qr
    #     qs = qs[:n+1]

    #     log_trans_probs = -np.log(1 + np.exp(-qs))
    #     success_prob = np.exp(np.cumsum(log_trans_probs))
    #     indep_prob = success_prob * (1 - np.exp(log_trans_probs))
    #     indep_prob = np.append(indep_prob, 1-np.sum(indep_prob))  # add total prob of success
    #     return np.argmax(np.random.multinomial(1, indep_prob))

    def _sample_transition(self, state, action):
        qr = np.array(state)
        n = action

        # for _ in range(self.T):
        #     fail_idx = self._sample_run(n, qr)
        #     qr = self._update_qr(n, qr, fail_idx)
        student = Student(lr=self.student_lr, q_e=self.eps)
        student.q_r = qr
        student.learn(BinaryEnv(n, reward=self.student_reward), max_iters=self.T)
        qr = student.q_r

        is_done = False
        reward = 0
        qs = self.student_qe + qr
        log_trans_probs = -np.log(1 + np.exp(-qs))
        log_success_prob = np.sum(log_trans_probs)
        if log_success_prob > -self.p_eps and n == self.goal_length:
            is_done = True
            reward = 10  # TODO: parameterize

        qr = tuple(self._round(qr))
        return qr, reward, is_done

    def _sample_rollout_policy(self, history):
        return np.random.choice(self.actions)   # TODO: use something better?
    
    def _init_node(self):
        return {'v': 0, 'n': 0}
    
    def _search(self):
        for _ in range(self.n_iters):
            self._simulate(self.history, 0)
        
        vals = [self.tree[self.history + (a,)]['v'] for a in self.actions]
        print('VALS', vals)
        return np.argmax(vals) + 1
    
    def _simulate(self, history, depth):
        reward_stack = []
        node_stack = []
        n_visited_stack = []

        while self.gamma ** depth > self.eps:
            if history not in self.tree:
                self.tree[history] = self._init_node()
                for a in self.actions:
                    proposal = history + (a,)
                    self.tree[proposal] = self._init_node()
                pred_reward = self._rollout(history, depth)
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

            a = np.argmax(vals) + 1
            # print('PROPOSED A', a)
            state = self.history[-1]
            next_state, reward, is_done = self._sample_transition(state, a)
            reward_stack.append(reward)

            self.tree[history]['n'] += 1
            next_node = self.tree[history + (a,)]
            next_node['n'] += 1
            node_stack.append(next_node)
            n_visited_stack.append(next_node['n'])

            history += (a, next_state)
            # print('NEW HIST', history)
            state = next_state
            depth += 1

            if is_done:
                break

        
        # backprop rewards
        for i, (node, n_visited) in enumerate(zip(node_stack, n_visited_stack)):
            total_reward = np.sum([r * self.gamma ** iters for iters, r in enumerate(reward_stack[i:])])
            node['v'] += (total_reward - node['v']) / n_visited


    def _rollout(self, history, depth):
        g = 1
        total_reward = 0

        state = history[-1]
        while self.gamma ** depth > self.eps:
            a = self._sample_rollout_policy(history)
            state, reward, is_done = self._sample_transition(state, a)

            history += (a, state)
            total_reward += g * reward
            g *= self.gamma
            depth += 1

            if is_done:
                break
        
        return total_reward
    
    def learn(self, env, is_eval=False,
             max_iters=1000, 
             use_tqdm=False, 
             post_hook=None, done_hook=None):
        raise NotImplementedError('TeacherPerfectKnowledge does not need to learn - it is cosmically perfect')
    
# <codecell>
# N = 3
# T = 50
# lr = 0.01
# reward = 10
# eps = 0


# teacher = TeacherPerfectKnowledge(goal_length=N, T=T, student_lr=lr, student_reward=reward, student_qe=eps)
# env = CurriculumEnv(goal_length=N, train_iter=T, student_qe_dist=eps, teacher_reward=10, student_reward=10, student_params={'lr': lr}, anarchy_mode=True)
# env.reset()

# qr = np.zeros(N)
# prev_a = None
# is_done = False

# while not is_done:
#     a = teacher.next_action(prev_a, qr)
#     state, reward, is_done, _ = env.step(a)

#     print(f'action: {a}   state: {state}')

#     prev_a = a
#     qr = np.array([env.student.q_r[i] for i in range(N)])

# print('Great success!')

# student = Student(lr=lr, q_e=eps)

# student.learn(BinaryEnv(N, reward=reward), max_iters=T)
# state, reward, is_done = teacher._sample_transition(np.zeros(N), N)

# print(state)
# print(student.q_r)

# %%
