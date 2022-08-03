"""
Simple binary environments for experimenting with teacher-student interactions

author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>
import copy

from collections import defaultdict
import numbers
import gym
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

def sig(x):
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
    def __init__(self, goal_length=10, train_iter=50, train_round=None,
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
        self.train_round = train_round
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

        trans = []
        def _update_trans(_, reward):
            result = int(reward > 0)
            trans.append(result)

        self.student.learn(BinaryEnv(self.N, reward=self.student_reward), max_iters=self.train_iter, max_rounds=self.train_round, done_hook=_update_trans)
        log_prob = self._get_score(self.N)
        reward = 0
        is_done = False

        if self.N == self.goal_length and -log_prob < self.p_eps:
            reward = self.teacher_reward
            is_done = True

        return (self.N, log_prob), reward, is_done, {'transcript': trans}
    
    def reset(self):
        self.student = Student(q_e=self.student_qe_dist, **self.student_params)
        student_score = self._get_score(self.goal_length, train=False)
        self.N  = 1
        return (self.N, student_score)

    def _get_score(self, length, train=True):
        # if train:
        #     self.student.learn(BinaryEnv(length, reward=self.student_reward), max_iters=self.train_iter)
        return self.student.score(length)


class Agent:
    def __init__(self) -> None:
        self.iter = 0

    def next_action(self, state):
        raise NotImplementedError('next_action not implemented in Agent')

    def update(old_state, action, reward, next_state, is_done):
        raise NotImplementedError('update not implemented in Agent')
    
    def learn(self, env, is_eval=False,
             max_iters=1000, 
             max_rounds=None,
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

                if max_rounds != None:
                    max_rounds -= 1
                    if max_rounds == 0:
                        break
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
        if isinstance(q_e, numbers.Number):
            self.q_e = defaultdict(lambda: q_e)
        elif type(q_e) != type(None):
            self.q_e = q_e
        else:
            self.q_e = defaultdict(int)
        self.q_r = defaultdict(int)
    
    # softmax policy
    def policy(self, state) -> np.ndarray:
        q = self.q_e[state] + self.q_r[state]
        prob = sig(q)
        return np.array([1 - prob, prob])

    def next_action(self, state) -> int:
        _, prob = self.policy(state)
        a = np.random.binomial(n=1, p=prob)
        return a 
    
    def update(self, old_state, _, reward, next_state, is_done):
        if is_done:
            exp_q = 0
            if reward == 0:  # null update to failing q(*,0) value
                return
        else:
            _, prob = self.policy(next_state)
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
                       student_lr=0.01, student_reward=10, 
                       n_particles=500, gamma=0.9, eps=1e-2, 
                       explore_factor=1, q_reinv_scale=1.5, q_reinv_prob=0.25) -> None:
        super().__init__()
        self.goal_length = goal_length
        self.T = T
        self.bins = bins
        self.p_eps = p_eps
        self.lookahead_cap = lookahead_cap
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
        self.qes_means = []
        self.qes_stds = []
        self.num_particles = []
        self.replicas = []
    
    
    def reset(self):
        self.history = ()
        self.tree = {}

        self.qrs_means = []
        self.qrs_stds = []
        self.num_particles = []
        self.curr_n = 1
    

    def next_action(self, prev_action=None, obs=None):
        if obs != None and prev_action != None:
            print(f'Observed: {obs}')
            self.history += (prev_action, obs,)

            qs = [(state[1], state[2]) for state in self.tree[self.history]['b']]
            qrs, qes = zip(*qs)
            qrs_mean = np.mean(qrs, axis=0)
            qrs_std = np.std(qrs, axis=0)
            qes_mean = np.mean(qes)
            qes_std = np.std(qes)

            self.num_particles.append(len(qrs))
            self.qrs_means.append(qrs_mean)
            self.qrs_stds.append(qrs_std)
            self.qes_means.append(qes_mean)
            self.qes_stds.append(qes_std)
            print('N_particles:', len(qrs))

        a = self._search()
        self.curr_n = np.clip(self.curr_n + a - 1, 1, self.goal_length)  # NOTE: assuming agent follows the proposed action
        return a
    
    def _sample_transition(self, state, action):
        n, qr, qe = state
        new_n = np.clip(n + action - 1, 1, self.goal_length)
        for _ in range(self.T):
            fail_idx = self._sim_fail(new_n, qr + qe)
            qr = self._update_qr(new_n, qr, qe, fail_idx)
        
        reward = 0
        is_done = False
        if -np.sum(np.log(self._sig(qr + qe))) < self.p_eps and new_n == self.goal_length:
            is_done = True
            reward = 10

        log_prob = np.sum([-np.log(1 + np.exp(-q)) for q in (qr + qe)[:new_n]])
        obs = self._to_bin(log_prob)
        
        return (new_n, qr, qe), obs, reward, is_done

    def _sim_fail(self, n, qs):
        for fail_idx, q in enumerate(qs[:n]):
            if self._sig(q) < np.random.random():
                return fail_idx
        
        return n

    def _update_qr(self, n, qr, qe, fail_idx):
        qr = np.copy(qr)
        if fail_idx == n:
            payoff = self.student_reward
        else:
            payoff = self._sig(qr[fail_idx] + qe) * qr[fail_idx]
        
        probs = self._sig(qr[1:fail_idx] + qe)
        rpe = np.append(probs * qr[1:fail_idx], payoff) - qr[:fail_idx]
        qr[:fail_idx] += self.student_lr * rpe
        return qr
    
    def _sig(self, x):
        return 1 / (1 + np.exp(-np.array(x)))

    def _to_bin(self, log_p, logit_min=-2, logit_max=2, eps=1e-8):
        logit = log_p - np.log(1 - np.exp(log_p) + eps)

        norm = (logit - logit_min) / (logit_max - logit_min)
        bin_p = np.clip(np.round(norm * self.bins), 0, self.bins)
        
        return bin_p

    def _sample_prior(self):
        qr = np.zeros(self.goal_length)
        qe = np.random.uniform(-5, 5)
        return (1, qr, qe) 

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
            # qrs = [state[1] for state in self.tree[self.history]['b']]
            # qrs_mean = np.mean(qrs, axis=0)
            # qrs_cov = self.q_reinv_scale * np.cov(qrs, rowvar=False)

            print('ITER WITH HIST', self.history)
            iters = []
            vict_iters = []
            for _ in range(self.n_particles):
                state_idx = np.random.choice(len(self.tree[self.history]['b']))
                state = self.tree[self.history]['b'][state_idx]

                # TODO: rethink reinvig strategy (jitter eps?)
                # if np.random.random() < self.q_reinv_prob:
                #     new_qrs = np.random.multivariate_normal(qrs_mean, qrs_cov)
                #     samp_state = self.tree[self.history]['b'][state_idx]   # randomly select fixed attributes
                #     state = (samp_state[0], new_qrs)
                # else:
                #     state = self.tree[self.history]['b'][state_idx]

                tot_iter, vict_iter = self._simulate(state, self.history, 0)
                iters.append(tot_iter)
                vict_iters.append(vict_iter)

            print('MEAN ITERS', np.mean(iters))
            print('N_ITERS', len(iters))
            print('PROP VICT', np.mean(vict_iters))
        
        vals = [self.tree[self.history + (a,)]['v'] for a in self.actions]
        print('VALS', vals)
        return np.argmax(vals)
    
    def _simulate(self, state, history, depth):
        reward_stack = []
        node_stack = []
        n_visited_stack = []

        tot_iter = 0
        vict_iter = 0

        while self.gamma ** depth > self.eps:
            tot_iter += 1
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

            if depth > 0:   # NOTE: avoid re-adding encountered state
                self.tree[history]['b'].append(state)
            self.tree[history]['n'] += 1

            next_node = self.tree[history + (a,)]
            next_node['n'] += 1
            node_stack.append(next_node)
            n_visited_stack.append(next_node['n'])

            if is_done:
                vict_iter = 1
                break

            history += (a, obs)
            state = next_state
            depth += 1
        
        for i, (node, n_visited) in enumerate(zip(node_stack, n_visited_stack)):
            total_reward = np.sum([r * self.gamma ** iters for iters, r in enumerate(reward_stack[i:])])
            node['v'] += (total_reward - node['v']) / n_visited
        
        return tot_iter, vict_iter


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
    
    def _sample_transition(self, state, action):
        qr = np.array(state)
        n = action

        student = Student(lr=self.student_lr, q_e=self.student_qe)
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


class TeacherPerfectKnowledgeDp(Agent):
    def __init__(self, goal_length=3, train_iters=50, p_eps=0.05, gamma=0.9, reward=10, n_bins_per_q=100, n_round_places=1, student_params=None) -> None:
        super().__init__()
        self.student_params = {
            'lr': 0.1,
            'reward': 10,
            'eps': 0
        }

        if student_params != None:
            for key, val in student_params.items():
                self.student_params[key] = val

        self.N = goal_length
        self.T = train_iters
        self.p_eps = p_eps
        self.gamma = gamma
        self.reward = reward
        self.n_bins_per_q = n_bins_per_q
        self.n_round_places = n_round_places

        self.state_axis = np.round(
            np.linspace(0, self.student_params['reward'], n_bins_per_q), 
            n_round_places)
        self.states = self._combo(self.state_axis, self.N)

        rand_ints = np.random.randint(self.N, size=len(self.states)) + 1
        self.policy = {s:a for s, a in zip(self.states, rand_ints)}
        self.value = {s:0 for s in self.states}
        
    
    def _combo(self, axis, dims):
        states = [(s,) for s in axis]
        for _ in range(dims - 1):
            states = [ss + (s,) for ss in states for s in axis]

        return states

    def next_action(self, state):
        state = tuple(self._disc(state))
        return self.policy[state]
    
    def learn(self, max_iters=100, eval_iters=3, with_tqdm=True):
        iterator = range(max_iters)
        if with_tqdm:
            iterator = tqdm(iterator)

        for i in iterator:
            # print('Step', i)
            self._policy_eval(max_iters=eval_iters)
            is_stable = self._policy_improv()

            if is_stable:
                print(f'info: policy converged in {i+1} steps')
                return
        
        print('warn: policy never converged')
    
    def _policy_eval(self, max_iters=999, eps=1e-2):
        for _ in range(max_iters):
            max_change = 0

            # print('Eval sweep')
            # for s in tqdm(self.states):
            for s in self.states:
                old_val = self.value[s]
                new_val = self._compute_value(s, self.policy[s])
                self.value[s] = new_val
                max_change = max(max_change, np.abs(old_val - new_val))
            
            if max_change < eps:
                break

    def _policy_improv(self):
        is_stable = True
        # print('Improv sweep')
        # for s in tqdm(self.states):
        for s in self.states:
            old_a = self.policy[s]
            vals = [self._compute_value(s, a) for a in np.arange(self.N) + 1]
            best_a = np.argmax(vals) + 1
            self.policy[s] = best_a

            if best_a != old_a:
                is_stable = False
        
        return is_stable
            
    
    def _compute_value(self, state, action):
        logprobs = self._compute_prob(state, action)
        val = 0
        for next_state, logprob in [(k, v) for k, v in logprobs.items() if v != -np.inf]:
            state_logprobs = self._logsig(np.array(next_state) + self.student_params['eps'])
            if action == self.N and -np.sum(state_logprobs) < self.p_eps:
                update = self.reward
            else:
                update = self.gamma * self.value[next_state]
            val += np.exp(logprob) * update
        return val
    
    def _compute_prob(self, state, action):
        # logprobs = {s:-np.inf for s in self.states}
        logprobs = defaultdict(lambda: -np.inf)
        eps = self.student_params['eps']

        fail_axis = np.arange(action + 1)
        fails = self._combo(fail_axis, self.T)

        for run in fails:
            qr = np.array(state)
            cum_log_prob = 0
            for fail_idx in run:
                cum_log_prob += np.sum(self._logsig(qr[:fail_idx] + eps))
                if fail_idx < action:
                    cum_log_prob += np.log(1 - self._sig(qr[fail_idx] + eps))

                qr = self._update_qr(action, qr, fail_idx)
            
            qr = tuple(self._disc(qr))
            logprobs[qr] = np.logaddexp(logprobs[qr], cum_log_prob)
        
        return logprobs


    def _update_qr(self, n, qr, fail_idx):
        if fail_idx == n:
            payoff = self.student_params['reward']
        else:
            payoff = self._sig(qr[fail_idx] + self.student_params['eps']) * qr[fail_idx]
        
        probs = self._sig(qr[1:fail_idx] + self.student_params['eps'])
        rpe = np.append(probs * qr[1:fail_idx], payoff) - qr[:fail_idx]
        qr[:fail_idx] += self.student_params['lr'] * rpe
        return qr
    
    def _sig(self, val):
        return 1 / (1 + np.exp(-val))
    
    def _logsig(self, val):
        return -np.log(1 + np.exp(-val))
    
    def _disc(self, qr):
        bin_idx = np.round(qr / self.student_params['reward'] * (self.n_bins_per_q - 1)).astype('int')
        return self.state_axis[bin_idx]

    def update(old_state, action, reward, next_state, is_done):
        raise NotImplementedError('Use learn() to train agent directly')

