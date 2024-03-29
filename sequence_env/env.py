"""
Collection of environments and teachers for the sequence-learning task.

author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>
from collections import defaultdict
import itertools
import numbers

import gym
import numpy as np
from scipy.stats import beta
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
                 anarchy_mode=False,
                 return_transcript=False,
                 track_qs=False):
        super().__init__()

        self.student = None
        self.goal_length = goal_length
        self.train_iter = train_iter
        self.train_round = train_round
        self.p_eps = p_eps
        self.teacher_reward = teacher_reward
        self.student_reward = student_reward
        self.student_qe_dist = student_qe_dist
        self.track_qs = track_qs

        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(goal_length), 
            gym.spaces.Box(low=0, high=1, shape=(1,))))
        self.N = 1

        self.action_space = gym.spaces.Discrete(3)
        self.student_params = student_params if student_params != None else {}
        self.anarchy_mode = anarchy_mode
        self.return_transcript = return_transcript
    
    def step(self, action):
        if self.anarchy_mode:
            self.N = np.clip(action, 1, self.goal_length)
        else:
            d_length = action - 1
            self.N = np.clip(self.N + d_length, 1, self.goal_length)

        trans = []
        all_qs = []
        def _update_trans(_, reward):
            result = int(reward > 0)
            trans.append(result)

            if self.track_qs:
                qs = [self.student.q_r[i] for i in range(self.goal_length)]
                all_qs.append(qs)
            
        self.student.learn(BinaryEnv(self.N, reward=self.student_reward), max_iters=self.train_iter, max_rounds=self.train_round, done_hook=_update_trans)
        log_prob = self._get_score(self.N)
        reward = 0
        is_done = False

        if self.N == self.goal_length and -log_prob < self.p_eps:
            reward = self.teacher_reward
            is_done = True

        metric = log_prob if not self.return_transcript else trans
        return (self.N, metric), reward, is_done, {'transcript': trans, 'qs': all_qs}
    
    def reset(self):
        self.student = Student(q_e=self.student_qe_dist, **self.student_params)
        student_score = self._get_score(self.goal_length, train=False)
        self.N  = 1
        return (self.N, student_score)

    def _get_score(self, length, train=True):
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
        if max_rounds != None:
            iterator = itertools.count()
        if use_tqdm:
            iterator = tqdm(iterator)

        fail_idxs = []
        for _ in iterator:
            action = self.next_action(state)
            next_state, reward, is_done, _ = env.step(action)
            if not is_eval:
                self.update(state, action, reward, next_state, is_done)

            if post_hook != None:
                post_hook(self)

            if is_done:
                fail_idxs.append(next_state)
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
        
        return fail_idxs


class Student(Agent):
    def __init__(self, lr=0.05, q_e=None, n_step=1) -> None:
        super().__init__()
        self.lr = lr
        self.n_step = n_step

        # only track Q-values for action = 1, maps state --> value
        if isinstance(q_e, numbers.Number):
            self.q_e = defaultdict(lambda: q_e)
        elif callable(q_e):
            self.q_e = defaultdict(q_e)
        elif type(q_e) != type(None):
            self.q_e = q_e
        else:
            self.q_e = defaultdict(int)

        self.q_r = defaultdict(int)
        self.buffer = []
    
    # softmax policy
    def policy(self, state) -> np.ndarray:
        q = self.q_e[state] + self.q_r[state]
        prob = sig(q)
        return np.array([1 - prob, prob])

    def next_action(self, state) -> int:
        _, prob = self.policy(state)
        a = np.random.binomial(n=1, p=prob)
        return a 
    
    # NOTE: n-step SARSA simplified for sequence-learning setting
    def update(self, old_state, _, reward, next_state, is_done):
        self.buffer.append(old_state)

        if is_done:
            if reward == 0 and len(self.buffer) >= self.n_step:   # account for "updated" q_e
                self.buffer = self.buffer[1:]

            for state in self.buffer:
                self.q_r[state] += self.lr * (reward - self.q_r[state])
            
            self.buffer = []
        else:
            if len(self.buffer) == self.n_step:
                target_state = self.buffer[0]
                self.buffer = self.buffer[1:]

                _, prob = self.policy(next_state)
                exp_q = prob * self.q_r[next_state]
                self.q_r[target_state] += self.lr * (exp_q - self.q_r[target_state])

    def score(self, goal_state) -> float:
        qs = [self.q_e[s] + self.q_r[s] for s in range(goal_state)]
        log_prob = np.sum([-np.log(1 + np.exp(-q)) for q in qs])
        return log_prob


class TeacherExpIncremental(Agent):
    def __init__(self, tau=0.95, discount=0.8) -> None:
        super().__init__()
        self.tau = tau
        self.discount = discount
        self.avgs = []
    
    def next_action(self, state):
        _, trans = state
        self._consume_trans(trans)

        if self.avgs[-1] > self.tau:
            return 2
        return 1
    
    def reset(self):
        self.avgs = []

    def _consume_trans(self, trans):
        avg = self.avgs[-1] if len(self.avgs) > 0 else 0
        for x in trans:
            avg = (1 - self.discount) * x + self.discount * avg
        self.avgs.append(avg)


class TeacherBetaIncremental(Agent):
    def __init__(self, tau=0.95, conf=0.5, max_m_factor=3) -> None:
        super().__init__()
        self.tau = tau
        self.conf = conf

        self.all_trans = []
        self.n = 1

        p_eps = -np.log(tau)
        raw_min_m = np.log(1 - conf) / (-p_eps) - 1
        self.min_m = int(np.floor(raw_min_m))
        self.max_m = int(self.min_m * max_m_factor)

        self.avgs = []
        self.bounds = []

    def next_action(self, state):
        _, trans = state
        self.all_trans.extend(trans)

        if self.do_jump():
            self.all_trans = []
            return 2

        return 1
    
    def do_jump(self):
        max_k = 1 + min(self.max_m, len(self.all_trans))
        do_jump = False
        chosen_k = None

        for k in range(self.min_m, max_k):
            prob_good = self._get_prob_good(self.all_trans[-k:])
            if prob_good >= self.conf:
                do_jump = True
                chosen_k = k
        
        if chosen_k == None:
            chosen_k = self.min_m

        samps = self.all_trans[-chosen_k:]
        n_succ = np.sum(samps)
        n_fail = len(samps) - n_succ

        beta_mean = (n_succ + 1) / (len(samps) + 2)
        beta_bounds = beta.ppf((0.025, 0.25, 0.75, 0.975), a=n_succ+1, b=n_fail+1)

        self.avgs.append(beta_mean)
        self.bounds.append(beta_bounds)
        return do_jump
    
    def _get_prob_good(self, transcript, tau=None):
        if tau == None:
            tau = self.tau

        success = np.sum(transcript)
        total = len(transcript)
        prob_bad = beta.cdf(tau, a=success+1, b=total-success+1)
        return 1 - prob_bad


class TeacherExpAdaptive(Agent):
    def __init__(self, goal_length, tree, dec_to_idx, discrete=False, prop_inc=100, shrink_factor=0.65, grow_factor=1.5, discount=0.8):
        self.goal_length = goal_length
        self.tree = tree
        self.dec_to_idx = dec_to_idx
        if discrete:
            self.inc = 1
            self.shrink_factor = 1
            self.grow_factor = 1
        else:
            self.inc = prop_inc
            self.shrink_factor = shrink_factor
            self.grow_factor = grow_factor

        self.discount = discount
        self.avgs = []
    
    def idx_to_act(self, idx):
        inc_idx = idx // 3
        jump_idx = idx % 3

        if inc_idx == 1:
            self.inc *= self.shrink_factor
        elif inc_idx == 2:
            self.inc *= self.grow_factor
            
        if jump_idx == 0:
            return -self.inc
        elif jump_idx == 1:
            return 0
        elif jump_idx == 2:
            return self.inc
        
        return None

    def dec_to_inc(self, dec, curr_n):
        idx = self.dec_to_idx[dec]
        act = self.idx_to_act(idx)
        return np.clip(act + curr_n, 1, self.goal_length).astype(int)
    
    def next_action(self, state):
        curr_n, trans = state
        self._consume_trans(trans)

        if len(self.avgs) == 1:
            return self.inc
        
        avg, last_avg = self.avgs[-1], self.avgs[-2]
        dec = self.tree.decide([avg, avg - last_avg])
        return self.dec_to_inc(dec, curr_n)
    
    def _consume_trans(self, trans):
        avg = self.avgs[-1] if len(self.avgs) > 0 else 0
        for x in trans:
            avg = (1 - self.discount) * x + self.discount * avg
        self.avgs.append(avg)


class TeacherTree:
    def __init__(self, splits, decisions=None, n_feats=2, n_splits=2) -> None:
        if type(splits) != np.ndarray:
            splits = np.array(splits)
        if type(decisions) == type(None):
            decisions = np.arange(n_splits**n_feats)
        elif type(decisions) != np.ndarray:
            decisions = np.array(decisions)

        self.splits = splits.reshape(n_feats, n_splits - 1)

        self.decisions = decisions.reshape((n_splits,) * n_feats)
    
    def decide(self, feats):
        result = self.decisions
        for i, x in enumerate(feats):
            split = self.splits[i]
            dec_idx = np.sum(x > split)
            result = result[dec_idx]
        return result


class TeacherParticleIncremental(Agent):
    def __init__(self, goal_length, n_particles=1000, conf=0.5, reinv_scale=0.5, tau=0.95, T=3, lr=0.1, student_reward=10) -> None:
        super().__init__()

        self.goal_length = goal_length
        self.n_particles = n_particles
        self.T = T
        self.R = student_reward
        self.lr = lr
        self.N = 1

        self.reinv_scale = reinv_scale
        self.conf = conf
        self.tau = tau

        self.particles = []
        self.all_particles = []

        self.avgs = []
        self.bounds = []
    
    def next_action(self, state):
        self.N, trans = state

        if len(trans) == 0:
            return 1

        self._condition(trans)
        if self._do_jump():
            return 2
        
        return 1
    
    def _condition(self, trans):
        trans = tuple(trans)

        if len(self.particles) == 0:
            self.particles = [self._sample_prior() for _ in range(self.n_particles)]
            self.all_particles.append(self.particles[:])

        outcomes = [self._sample_transition(p) for p in self.particles]
        self.particles = np.array([p for p, t in outcomes if t == trans], dtype=object)

        # resample
        log_probs = np.array([self._log_prob(p, trans) for p in self.particles])
        raw_probs = np.exp(log_probs - np.max(log_probs))
        probs = raw_probs / np.sum(raw_probs)

        samp_idxs = np.random.choice(len(self.particles), replace=True, p=probs, size=self.n_particles)
        self.particles = self.particles[samp_idxs]
        self.particles = self._reinvigorate(self.particles)
        self.all_particles.append(np.copy(self.particles))

    def _do_jump(self):
        sn = np.array([self._get_success(p) for p in self.particles])
        conf_jump = np.mean(sn > self.tau)

        self.avgs.append(np.mean(sn))
        self.bounds.append(np.quantile(sn, (0.025, 0.25, 0.75, 0.975)))

        return conf_jump > self.conf

    def _get_success(self, params):
        qr, qe = params
        probs = self._sig(qr + qe)
        return np.prod(probs[:self.N])
    
    def _log_prob(self, params, trans):
        qr, qe = params
        prob = self._sig(qr + qe)
        prob = np.prod(prob[:self.N])

        n1 = np.sum(trans)
        n0 = len(trans) - n1

        return n0 * np.log(1 - prob) + n1 * np.log(prob)

    def _sample_transition(self, params):
        qr, qe = params
        new_n = self.N
        trans = []

        for _ in range(self.T):
            fail_idx = self._sim_fail(new_n, qr + qe)
            qr = self._update_qr(new_n, qr, qe, fail_idx)
            trans.append(int(fail_idx == new_n))
        
        return (qr, qe), tuple(trans)

    def _sim_fail(self, n, qs):
        for fail_idx, q in enumerate(qs[:n]):
            if self._sig(q) < np.random.random():
                return fail_idx
        
        return n

    def _update_qr(self, n, qr, qe, fail_idx):
        qr = np.copy(qr)
        if fail_idx == n:
            payoff = self.R
        else:
            payoff = self._sig(qr[fail_idx] + qe) * qr[fail_idx]
        
        probs = self._sig(qr[1:fail_idx] + qe)
        rpe = np.append(probs * qr[1:fail_idx], payoff) - qr[:fail_idx]
        qr[:fail_idx] += self.lr * rpe
        return qr
    
    def _sig(self, x):
        return 1 / (1 + np.exp(-np.array(x)))

    def _sample_prior(self):
        qr = np.zeros(self.goal_length)
        qe = np.random.uniform(-5, 5)
        return (qr, qe) 
    
    def _reinvigorate(self, particles):
        new_p = [(qr, eps + self.reinv_scale * np.random.randn()) for qr, eps in particles]
        return new_p


class TeacherPomcpAgentClean(Agent):
    def __init__(self, goal_length, T, bins=10, p_eps=0.05,
                       student_reward=10, 
                       n_particles=500, gamma=0.9, eps=1e-2, 
                       explore_factor=1, q_reinv_prob=0.25, q_reinv_scale=0.5) -> None:
        super().__init__()
        self.goal_length = goal_length
        self.T = T
        self.bins = bins
        self.p_eps = p_eps
        self.student_reward = student_reward
        self.q_reinv_prob = q_reinv_prob
        self.q_reinv_scale = q_reinv_scale

        self.n_particles = n_particles
        self.gamma = gamma
        self.eps = eps
        self.explore_factor = explore_factor

        self.actions = [0, 1, 2]
        self.history = ()
        self.full_history = ()
        self.tree = MctsTree()

        self.curr_n = 1
        self.qrs_means = []
        self.qrs_stds = []
        self.qes_means = []
        self.qes_stds = []
        self.lr_means = []
        self.lr_stds = []
        self.num_particles = []
    
    
    def reset(self):
        self.history = ()
        self.tree = MctsTree()

        self.qrs_means = []
        self.qrs_stds = []
        self.num_particles = []
        self.curr_n = 1
    

    def next_action(self, prev_action=None, obs=None):
        if type(obs) != type(None) and prev_action != None:
            print(f'Observed: {obs}')
            self.history += (prev_action, obs,)
            self.full_history += (prev_action, obs,)

            if self.history not in self.tree or len(self.tree[self.history]['b']) == 0:
                raise Exception('fail to converge')
            else:
                self.tree.reroot(self.history)
                self.history = self.history[-1:]

                qs = [(state[1], state[2], state[3]) for state in self.tree[self.history]['b']]
                qrs, qes, lrs = zip(*qs)
                qrs_mean = np.mean(qrs, axis=0)
                qrs_std = np.std(qrs, axis=0)
                qes_mean = np.mean(qes)
                qes_std = np.std(qes)
                lr_mean = np.mean(lrs)
                lr_std = np.std(lrs)

                self.num_particles.append(len(qrs))
                self.qrs_means.append(qrs_mean)
                self.qrs_stds.append(qrs_std)
                self.qes_means.append(qes_mean)
                self.qes_stds.append(qes_std)
                self.lr_means.append(lr_mean)
                self.lr_stds.append(lr_std)
                print('N_particles:', len(qrs))

        a = self._search()
        self.curr_n = np.clip(self.curr_n + a - 1, 1, self.goal_length)  # NOTE: assuming agent follows the proposed action
        return a
    
    def _sample_transition(self, state, action):
        n, qr, qe, lr = state
        new_n = np.clip(n + action - 1, 1, self.goal_length)
        trans = []

        for _ in range(self.T):
            fail_idx = self._sim_fail(new_n, qr + qe)
            qr = self._update_qr(new_n, qr, qe, lr, fail_idx)
            trans.append(int(fail_idx == new_n))
        
        reward = 0
        is_done = False
        if -np.sum(np.log(self._sig(qr + qe))) < self.p_eps and new_n == self.goal_length:
            is_done = True
            reward = 10

        return (new_n, qr, qe, lr), tuple(trans), reward, is_done

    def _sim_fail(self, n, qs):
        for fail_idx, q in enumerate(qs[:n]):
            if self._sig(q) < np.random.random():
                return fail_idx
        
        return n

    def _update_qr(self, n, qr, qe, lr, fail_idx):
        qr = np.copy(qr)
        if fail_idx == n:
            payoff = self.student_reward
        else:
            payoff = self._sig(qr[fail_idx] + qe) * qr[fail_idx]
        
        probs = self._sig(qr[1:fail_idx] + qe)
        rpe = np.append(probs * qr[1:fail_idx], payoff) - qr[:fail_idx]
        qr[:fail_idx] += lr * rpe
        return qr
    
    def _sig(self, x):
        return 1 / (1 + np.exp(-np.array(x)))

    def _sample_prior(self):
        qr = np.zeros(self.goal_length)
        qe = np.random.uniform(-5, 5)
        lr = np.random.uniform(0, 1)
        return (1, qr, qe, lr) 

    def _sample_rollout_policy(self, history):
        return np.random.choice(self.actions)
    
    def _sample_inc_policy(self, state):
        n, qr, qe = state[:3]
        total_log_prob = 0
        for i, q in enumerate(qr + qe):
            total_log_prob += np.log(self._sig(q))
            if total_log_prob < -self.p_eps:
                break
        
        if i + 1 < n:
            return 0
        elif i + 1 > n:
            return 2
        else:
            return 1

    def _init_node(self):
        return {'v': 0, 'n': 0, 'b': []}
    
    def _search(self):
        if len(self.history) == 0:
            for _ in range(self.n_particles):
                state = self._sample_prior()
                self._simulate(state, self.history, 0)
        else:
            curr_states = self.tree[self.history]['b']

            iters = []
            vict_iters = []
            for _ in range(self.n_particles):
                state_idx = np.random.choice(len(curr_states))
                state = curr_states[state_idx]

                state = (state[0], # N
                         state[1], # q_r
                         state[2] + self.q_reinv_scale * np.random.randn(), # eps
                         state[3]  # lr
                )

                tot_iter, vict_iter = self._simulate(state, self.history, 0)
                iters.append(tot_iter)
                vict_iters.append(vict_iter)
        
        children = self.tree._traverse(self.history).children
        vals = [children[a].value['v'] for a in self.actions]
        
        if np.all([v == 0 for v in vals]):
            raise Exception('no values converged')

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
                new_node = MctsNode(self._init_node())
                new_node.children = {a: MctsNode(self._init_node()) for a in self.actions}
                curr_node = self.tree._traverse(history[:-1])

                if len(history) > 0:
                    curr_node.children[history[-1]] = new_node
                else:
                    self.tree.root = new_node

                pred_reward = self._rollout(state, history, depth)
                reward_stack.append(pred_reward)
                break

            vals = []
            curr_node = self.tree._traverse(history)
            for a in self.actions:
                next_node = curr_node.children[a].value
                if curr_node.value['n'] > 0 and next_node['n'] > 0:
                    explore = self.explore_factor * np.sqrt(np.log(curr_node.value['n']) / next_node['n'])
                else:
                    explore = 999  # arbitrarily high

                vals.append(next_node['v'] + explore)

            a = np.argmax(vals)
            next_state, trans, reward, is_done = self._sample_transition(state, a)
            reward_stack.append(reward)

            if depth > 0:   # NOTE: avoid re-adding encountered state
                curr_node.value['b'].append(state)
            curr_node.value['n'] += 1

            next_node = curr_node.children[a].value
            next_node['n'] += 1
            node_stack.append(next_node)
            n_visited_stack.append(next_node['n'])

            if is_done:
                vict_iter = 1
                break

            history += (a, trans)
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
            # a = self._sample_rollout_policy(history)
            a = self._sample_inc_policy(state)
            state, obs, reward, is_done = self._sample_transition(state, a)

            history += (a, obs)
            total_reward += g * reward
            g *= self.gamma
            depth += 1

            if is_done:
                break
        
        return total_reward
    

class MctsTree:
    def __init__(self) -> None:
        self.root = MctsNode(None)
    
    def _traverse(self, key):
        if len(key) == 0:
            return self.root

        node = self.root[key[0]]
        for k in key[1:]:
            node = node[k]
        return node
        
    def __getitem__(self, key):
        return self._traverse(key).value
    
    def __setitem__(self, key, value):
        node = self._traverse(key[:-1])
        if node[key[-1]] != None:
            node[key[-1]].value = value
        else:
            node[key[-1]] = MctsNode(value)
    
    def __contains__(self, key):
        if len(key) == 0 and self.root.value == None:
            return False

        try:
            node = self._traverse(key)
        except:
            return False
        return node != None
    
    def reroot(self, key):
        new_root = self._traverse(key)
        self.root = MctsNode(None)
        self.root[key[-1]] = new_root
    
    def merge(self, other, merge_func):
        self.root.merge(other.root, merge_func)
    
    def __str__(self):
        return str(self.root)
    
    def __repr__(self):
        return str(self)


class MctsNode:
    def __init__(self, value) -> None:
        self.value = value
        self.children = {}
    
    def __getitem__(self, key):
        if key not in self.children:
            return None

        return self.children[key]
    
    def __setitem__(self, key, node):
        self.children[key] = node

    def __eq__(self, other):
        if other == None:
            return False

        return self.value == other.value
    
    def merge(self, other, merge_func):
        if self.value != None and other.value != None:
            self.value = merge_func(self.value, other.value)

        for key, child in other.children.items():
            if key in self.children:
                self.children[key].merge(child, merge_func)
            else:
                self.children[key] = child
    
    def __str__(self):
        children_str = '\n'.join([f'{key}: {str(child)}' for key, child in self.children.items()])
        return f'{self.value} -> [{children_str}]'
    
    def __repr__(self):
        return str(self)
