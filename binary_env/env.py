"""
Simple binary environments for experimenting with teacher-student interactions

author: William Tong (wtong@g.harvard.edu)
"""
# <codecell>
from collections import defaultdict
import itertools
from multiprocessing import Pool

import numbers
import warnings
import gym
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import beta
from sklearn.metrics.pairwise import rbf_kernel
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
    
    # NOTE: specially adapted to binary env setting (deviates from vanilla n-step Sarsa)
    def update(self, old_state, _, reward, next_state, is_done):
        self.buffer.append(old_state)

        if is_done:
            # NOTE: removed for simplification <-- does it actually help?
            # NOTE: confirm still works for discrete case v
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


class TeacherUncertainOsc(Agent):
    def __init__(self, goal_length, tau=0.95, conf=0.2, max_m_factor=3, with_backtrack=False, bt_tau=0.25, bt_conf=0.2) -> None:
        super().__init__()
        self.goal_length = goal_length
        self.tau = tau
        self.conf = conf
        self.with_backtrack = with_backtrack
        self.bt_tau = bt_tau
        self.bt_conf = bt_conf

        self.trans_dict = defaultdict(list)
        self.n = 1

        p_eps = -np.log(tau)
        raw_min_m = np.log(1 - conf) / (-p_eps) - 1
        self.min_m = int(np.floor(raw_min_m))
        self.max_m = int(self.min_m * max_m_factor)

    def next_action(self, state):
        curr_n, trans = state
        self.trans_dict[curr_n].extend(trans)

        next_n = self.n
        if self.do_jump():
            self.n = min(self.n + 1, self.goal_length)
            next_n = self.n
        elif self.with_backtrack and self.do_dive():
            self.n = max(self.n - 1, 1)
            next_n = self.n
        elif next_n == curr_n:
            next_n = max(self.n - 1, 1)
        
        return next_n
    
    def do_jump(self):
        trans = self.trans_dict[self.n]
        for k in range(self.min_m, 1 + min(self.max_m, len(trans))):
            prob_good = self._get_prob_good(trans[-k:])
            if prob_good >= self.conf:
                return True

        return False

    def do_dive(self):
        if self.n == 1:
            return

        trans = self.trans_dict[self.n - 1]
        rev_trans = [not bit for bit in trans]
        for k in range(self.min_m, 1 + min(self.max_m, len(trans))):
            prob_good = self._get_prob_good(rev_trans[-k:], 1 - self.bt_tau)
            if prob_good >= self.bt_conf:
                return True
        
        return False
    
    def _get_prob_good(self, transcript, tau=None):
        if tau == None:
            tau = self.tau

        success = np.sum(transcript)
        total = len(transcript)
        prob_bad = beta.cdf(tau, a=success+1, b=total-success+1)
        return 1 - prob_bad


# TODO: clean up and work out rigorous tuning
class TeacherAdaptive(Agent):
    def __init__(self, goal_length, threshold=0.95, threshold_low=0.05, tau=0.5, conf=0.95, max_m_factor=3, abs_min_m=5, cut_factor=2, student=None, with_osc=False) -> None:
        super().__init__()
        self.goal_length = goal_length
        self.threshold = threshold
        self.threshold_low = threshold_low
        self.tau = tau
        self.conf = conf
        self.student = student
        self.with_osc = with_osc

        p_eps = -np.log(threshold)
        raw_min_m = int(np.round(np.log(1 - conf) / (-p_eps) - 1))
        self.min_m = max(raw_min_m, abs_min_m)
        self.max_m = self.min_m * max_m_factor

        self.cut_factor = cut_factor
        # self.prop_inc = goal_length // cut_factor
        self.prop_inc = 100
        self.inc = None
        self.transcript = []
        self.in_osc = False
    
    def next_action(self, state):
        curr_n, trans = state
        if self.in_osc:
            self.in_osc = False
            return int(curr_n + self.inc)
        else:
            self.transcript.extend(trans)

            if self.inc != None:   # incremental
                next_n = curr_n
                if len(self.transcript) > self.min_m:
                    if self.do_jump():
                        next_n = min(curr_n + self.inc, self.goal_length)
                        return int(next_n)
                    elif self.do_dive():
                        next_n //= self.cut_factor
                        self.inc = max(self.inc // 2, 1)
                        return int(next_n)

                if self.with_osc:
                    self.in_osc = True
                    return int(curr_n - self.inc)

                return int(next_n)

            elif len(self.transcript) > (self.min_m + self.max_m) // 2:   # binary search
                next_n = self.prop_inc

                if self.do_jump(thresh=self.tau):
                    self.inc = self.prop_inc
                    next_n = min(curr_n + self.inc, self.goal_length)
                    self.transcript = []
                else:
                    self.prop_inc //= self.cut_factor
                    next_n = self.prop_inc
                    self.transcript = []

                return int(next_n)
            
            return int(self.prop_inc)
            
    def do_jump(self, trans=None, thresh=None):
        trans = self.transcript if trans == None else trans
        for k in range(self.min_m, 1 + min(self.max_m, len(trans))):
            prob_good = self._get_prob_good(trans[-k:], thresh=thresh)
            if prob_good >= self.conf:
                return True

        return False

    def do_dive(self):
        rev_trans = [not bit for bit in self.transcript]
        return self.do_jump(rev_trans, 1 - self.threshold_low)

    def _get_prob_good(self, transcript, thresh=None):
        if thresh == None:
            thresh = self.threshold

        success = np.sum(transcript)
        total = len(transcript)
        prob_bad = beta.cdf(thresh, a=success+1, b=total-success+1)
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
        if decisions != None and type(decisions) != np.ndarray:
            decisions = np.array(decisions)

        self.splits = splits.reshape(n_feats, n_splits - 1)
        if decisions == None:
            decisions = np.arange(n_feats * n_splits)

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

        # print('N_PARTS', len(self.particles))

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

            # lr = [state[3] for state in curr_states]
            # lr_bounds = (np.min(lr), np.max(lr))

            print('ITER WITH HIST', self.full_history)
            iters = []
            vict_iters = []
            for _ in range(self.n_particles):
                state_idx = np.random.choice(len(curr_states))
                state = curr_states[state_idx]

                # if np.random.random() < self.q_reinv_prob:
                #     new_eps = state[2] + self.q_reinv_scale * self.random.randn()
                #     new_lr = np.random.uniform(*lr_bounds)
                #     state = (state[0], state[1], new_eps, new_lr)

                state = (state[0], # N
                         state[1], # q_r
                         state[2] + self.q_reinv_scale * np.random.randn(), # eps
                         state[3]  # lr
                )

                tot_iter, vict_iter = self._simulate(state, self.history, 0)
                iters.append(tot_iter)
                vict_iters.append(vict_iter)

            print('MEAN ITERS', np.mean(iters))
            print('N_ITERS', len(iters))
            # print('PROP VICT', np.mean(vict_iters))
        
        children = self.tree._traverse(self.history).children
        vals = [children[a].value['v'] for a in self.actions]
        
        print('VALS', vals)
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

        
# tree = MctsTree()
# tree[('a',)] = 1
# tree[('a', 'b',)] = 2
# tree[('a', 'b_')] = 3
# print(str(tree))

# tree2 = MctsTree()
# tree2[('a',)] = 1
# tree2[('a', 'b__',)] = 20
# tree2[('a', 'b_')] = 30

# tree.merge(tree2, lambda a,b: a + b)
# print(str(tree))

class TeacherMctsCont(Agent):
    def __init__(self, N_eff, update_width=100, T=5, threshold=0.95, bandwidth=10, student_params=None, n_iters=1000, gamma=0.9, pw_init=5, pw_alpha=0.8, explore_factor=1, n_jobs=16) -> None:
        super().__init__()
        self.N_eff = N_eff
        self.T = T
        self.update_width = update_width
        self.threshold = threshold
        self.bandwidth = bandwidth
        self.student_params = {
            'lr': 0.1,
            'eps_eff': 0,
            'reward': 10
        }

        if student_params != None:
            for key, val in student_params.items():
                self.student_params[key] = val
        
        self.n_iters = n_iters
        self.gamma = gamma
        self.pw_init = pw_init
        self.pw_alpha = pw_alpha
        self.explore_factor = explore_factor
        self.n_jobs = n_jobs

        self.N, self.eps = TeacherMctsCont._to_cont(self.N_eff, self.student_params['eps_eff'])
        self.actions = np.arange(self.N) + 1
        self.actions_rand = np.random.permutation(self.N-1) + 1
        self.actions_rand = np.append(self.N, self.actions_rand)  # ensure goal length is always present

        self.history = ()
        self.tree = MctsTree()

    @staticmethod
    def _to_cont(N_eff, eps_eff, dn_per_interval=100):
        prob = sig(eps_eff) ** (1 / dn_per_interval)
        eps_cont = np.log(prob / (1 - prob))
        N_cont = N_eff * dn_per_interval

        return N_cont, eps_cont

    def reset(self):
        self.history = ()
        self.tree = MctsTree()
    
    def next_action(self, prev_action=None, qr=None):
        self.iter += 1

        if prev_action != None and type(qr) != None:
            qr = self._round(qr)
            self.history += (prev_action, tuple(self._round(qr)))
            if self.history not in self.tree:
                print('warn: rerooting tree')
                self.tree = MctsTree()
                self.history = self.history[-1:]
            else:
                self.tree.reroot(self.history)
                self.history = self.history[-1:]
        else:
            self.history = (tuple(np.zeros(self.N)),)
            
        print('ITER', self.iter)
        pw_size = np.ceil(self.pw_init * (self.iter) ** self.pw_alpha).astype(int)
        actions = self.actions_rand[:pw_size]

        print('PW_SIZE', pw_size)
        print('ACTIONS', actions)

        args = {
            'tree': self.tree,
            'history': self.history,
            'bandwidth': self.bandwidth,
            'actions': actions,
            'n_particles': self.n_iters,
            'N_cont': self.N,
            'eps_cont': self.eps,
            'threshold': self.threshold,
            'T': self.T,
            'lr': self.student_params['lr'],
            'update_width': self.update_width,
            'gamma': self.gamma,
            'eps_end': 0.01,
            'explore_factor': 1
        }

        # all_args = [copy.deepcopy(args) for _ in range(self.n_jobs)]
        all_args = [args for _ in range(self.n_jobs)]

        with Pool(self.n_jobs) as p:
            trees = p.map(_mcts_search, all_args)
        
        all_a = []
        all_vals = []
        for t in trees:
            vals = [t[self.history + (a,)]['v'] for a in self.actions]
            action = np.argmax(vals) + 1
            all_a.append(action)
            all_vals.append(vals[action-1])
        
        print('ALL_A', all_a)
        print('ALL_V', all_vals)
        # self.trees = trees
        a_ker = _rbf_kernel(all_a, self.bandwidth)
        votes = a_ker @ np.array(all_vals).reshape(-1, 1)
        print('VOTES', votes)
        best_idx = np.argmax(votes.flatten())
        a = all_a[best_idx]

        self.tree = self._merge_trees(trees)
        return a
    
    def _merge_trees(self, trees):
        total_tree = trees[0]

        def merge_func(n1, n2):
            total_n = n1['n'] + n2['n']
            total_v = 0
            if total_n != 0:
                total_v = (n1['n'] * n1['v'] + n2['n'] * n2['v']) / total_n
            return {'n': total_n, 'v': total_v}

        for t in trees[1:]:
            total_tree.merge(t, merge_func)

        return total_tree
    
    def _round(self, val):
        return np.round(val, decimals=1)

def _rbf_kernel(xs, bandwidth):
    xs = np.array(xs).reshape(-1, 1)
    gamma = 1 / bandwidth ** 2
    return rbf_kernel(xs, gamma=gamma)

def _mcts_search(params):
    tree = params['tree']
    history = params['history']
    actions = params['actions']
    bandwidth = params['bandwidth']
    n_particles = params['n_particles']
    N_cont = params['N_cont']
    eps_cont = params['eps_cont']
    threshold = params['threshold']
    T = params['T']
    lr = params['lr']
    update_width = params['update_width']
    gamma = params['gamma']
    eps_end = params['eps_end']
    explore_factor = params['explore_factor']

    np.random.seed()   # reset seed from parent
    K = _rbf_kernel(actions, bandwidth)

    def _init_node():
        return {'v': 0, 'n': 0}
    
    def _rollout(history, depth):
        g = 1
        total_reward = 0

        while gamma ** depth > eps_end:
            state = history[-1]
            a = _sample_inc_policy(state)
            state, reward, is_done = _sample_transition(state, a)

            history += (a, state)
            total_reward += g * reward
            g *= gamma
            depth += 1

            if is_done:
                break
        
        return total_reward

    def _sample_inc_policy(state):
        qr = state
        total_log_prob = 0
        qs = np.array(qr) + eps_cont
        for i, q in enumerate(qs):
            total_log_prob += np.log(sig(q))
            if total_log_prob < -0.05: # TODO: hardcoded
                break
        
        return i + 1

    def _sample_transition(state, action):
        qr = np.array(state)
        n = action

        for _ in range(T):
            fail_idx = _sim_fail(n, qr + eps_cont)
            qr = _update_qr(n, qr, eps_cont, lr, fail_idx)
        
        reward = 0
        is_done = False
        if np.exp(np.sum(np.log(sig(qr + eps_cont)))) > threshold and n == N_cont:
            is_done = True
            reward = 10
        
        qr = tuple(_round(qr))
        return qr, reward, is_done

    def _sim_fail(n, qs):
        for fail_idx, q in enumerate(qs[:n]):
            if sig(q) < np.random.random():
                return fail_idx
        
        return n

    def _update_qr(n, qr, qe, lr, fail_idx):
        qr = np.copy(qr).astype('float')
        if fail_idx == n:
            payoff = 10  # TODO: hardcoded
        else:
            payoff = 0
        
        probs = sig(qr[update_width:fail_idx] + qe)
        exp_return = np.append(probs * qr[update_width:fail_idx], np.repeat(payoff, min(update_width, fail_idx)))

        qr[:fail_idx] += lr * (exp_return - qr[:fail_idx])
        return qr
    
    def _round(val):
        return np.round(val, decimals=1)

    def _simulate(history):
        reward_stack = []
        node_stack = []
        n_visited_stack = []
        depth = 0

        while gamma ** depth > eps_end:
            if history not in tree:
                new_node = MctsNode(_init_node())
                new_node.children = {a: MctsNode(_init_node()) for a in np.arange(N_cont) + 1}
                curr_node = tree._traverse(history[:-1])
                curr_node.children[history[-1]] = new_node

                pred_reward = _rollout(history, depth)
                reward_stack.append(pred_reward)
                break
            
            children = tree._traverse(history).children
            all_next_nodes = [children[a].value for a in actions]
            results = np.array([(n['v'], n['n']) for n in all_next_nodes])
            exp_val = K @ (results[:,0] * results[:,1]).reshape(-1, 1)
            visits = K @ results[:,1].reshape(-1, 1) + 1e-8

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                vals = exp_val + explore_factor * np.sqrt(np.log(np.sum(visits)) / visits)

            a = actions[np.argmax(vals)]
            state = history[-1]
            next_state, reward, is_done = _sample_transition(state, a)
            reward_stack.append(reward)

            tree[history]['n'] += 1
            next_node = tree[history + (a,)]
            next_node['n'] += 1
            node_stack.append(next_node)
            n_visited_stack.append(next_node['n'])

            history += (a, next_state)
            state = next_state
            depth += 1

            if is_done:
                break

        # backprop rewards
        for i, (node, n_visited) in enumerate(zip(node_stack, n_visited_stack)):
            total_reward = np.sum([r * gamma ** iters for iters, r in enumerate(reward_stack[i:])])
            node['v'] += (total_reward - node['v']) / n_visited

    '''
    body of overall function
    '''
    for _ in range(n_particles):
        _simulate(history)

    return tree
    

if __name__ == '__main__':
    teacher = TeacherMctsCont(10, n_jobs=48, n_iters=50, pw_init=5, gamma=0.97)
    # teacher.next_action()

    env = CurriculumEnv(goal_length=teacher.N, train_iter=999, train_round=5, p_eps=0.05, teacher_reward=10, student_reward=10, student_qe_dist=teacher.eps, student_params={'lr': 0.1, 'n_step':100}, anarchy_mode=True)
    traj = [env.N]
    env.reset()
    prev_qr = None
    prev_a = None

    for _ in range(1000):
        a = teacher.next_action(prev_a, prev_qr)
        print('TOOK ACTION', a)

        state, _, is_done, _ = env.step(a)
        traj.append(a)

        prev_a = a
        prev_qr = [env.student.q_r[i] for i in range(teacher.N)]

        if is_done:
            break

    print('done!')
    plt.plot(traj)
    plt.savefig('traj.png')

# <codecell>
# # TODO: validate closeness of teacher's model and student
# N_cont, eps_cont = TeacherMctsCont._to_cont(3, 0)

# action = 300
# init_qrs = np.zeros(N_cont)

# pred_qrs, _, _ = _sample_transition(tuple(init_qrs), action)

# student = Student(lr, eps_cont, n_step=update_width)
# student.q_r = np.copy(init_qrs)
# fail_idxs = student.learn(BinaryEnv(action, reward=10), max_iters=99999, max_rounds=T)
# print('FAIL IDXS', fail_idxs)

# # qrs = np.copy(init_qrs)
# # for idx in fail_idxs:
# #     qrs = _update_qr(action, qrs, eps_cont, lr, idx)


# # pred_qrs = np.array(pred_qrs)
# # print('PRED', pred_qrs[:150])
# print('TRUE', student.q_r[:150])
# print('PRED', np.array(pred_qrs[:150]))
    
    
# %%
