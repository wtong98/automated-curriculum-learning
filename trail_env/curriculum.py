"""
Training schedules

author: William Tong (wtong@g.harvard.edu)
"""

from collections import defaultdict
from itertools import chain, zip_longest
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import beta

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv 

from env import TrailEnv
from trail_map import MeanderTrail

class CurriculumCallback(BaseCallback):
    def __init__(self, teacher, eval_env=None, save_every=0, save_path='trained', verbose=0, next_lesson_callbacks=None):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.save_every = save_every
        self.teacher = teacher
        self.next_lesson_callbacks = next_lesson_callbacks or []

        self.save_path = Path(save_path)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)

        self.curr_iter = 0
        self.curr_gen = 0
    
    def _on_training_start(self) -> None:
        self.teacher.load_logger(self.logger)
        self.teacher.load_student(self.model, self.eval_env)
        self.teacher.load_training_env(self.training_env)
        self.curr_ckpt = self.teacher.next_checkpoint()
        self.training_env.env_method('queue_map', self.curr_ckpt['map'])
        self.eval_env.env_method('queue_map', self.curr_ckpt['map'])
        
        for cb in self.next_lesson_callbacks:
            cb(self)
    
    def _on_rollout_end(self) -> None:
        print('STARTING UPDATE')
        return super()._on_rollout_end()

    def _on_step(self) -> bool:
        self.curr_iter += 1
        if self.curr_iter > self.curr_ckpt['iters']:
            print('HITTING CHECKPOINT')
            for cb in self.next_lesson_callbacks:
                cb(self)

            try:
                self.curr_ckpt = self.teacher.next_checkpoint()
            except StopIteration:
                return False

            self.training_env.env_method('queue_map', self.curr_ckpt['map'])
            self.eval_env.env_method('queue_map', self.curr_ckpt['map'])
            self.curr_iter = 0

            self.curr_gen += 1
            if self.save_every > 0 and self.curr_gen % self.save_every == 0:
                self.model.save(str(self.save_path / f'gen{self.curr_gen}'))

        return True
    
    def _on_training_end(self) -> None:
        if self.save_path:
            self.model.save(str(self.save_path / 'gen_final'))



class ManualTeacher:
    def __init__(self, trail_class):
        self.trail_class = trail_class
        self.checkpoints = []
        self.ckpt_idx = 0
    
    def add_ckpt(self, iters, **trail_args):
        self.checkpoints.append({
            'iters': iters,
            'map': self.trail_class(**trail_args)
        })

        return self
    
    def load_student(self, student, eval_env=None):
        pass   # do nothing
    
    def next_checkpoint(self):
        if self.ckpt_idx >= len(self.checkpoints):
            raise StopIteration

        next_ckpt = self.checkpoints[self.ckpt_idx]
        self.ckpt_idx += 1
        return next_ckpt

    def __iter__(self):
        return iter(self.checkpoints)


class Teacher:
    def __init__(self, n_iters_per_ckpt=1000, sched=None, trail_class=None):
        self.trail_class = trail_class if trail_class != None else MeanderTrail

        self.sched_idx = 0
        if type(sched) == type(None):
            self.sched = lambda x: x
        elif hasattr(sched, '__getitem__'):
            self.sched = lambda x: sched[x]
            self.sched_len = len(sched)
        else:
            self.sched = sched

        self.n_iters_per_ckpt = n_iters_per_ckpt
        self.n_test_episodes = 5

        self.student = None
        self.eval_env = None
        self.fresh = True
        self.trajectory = []
        self.logger = None
        self.training_env = None
    
    def load_logger(self, logger):
        self.logger = logger

    def load_student(self, student, eval_env):
        self.student = student
        self.eval_env = eval_env
        self.fresh = True
        self.trajectory = []
        self.history = defaultdict(list)
        self.trans = None
    
    def load_training_env(self, env):
        self.training_env = env
    
    def next_checkpoint(self):
        if self.student == None or self.eval_env == None:
            raise Exception('student or eval_env not initialized: load_student() with student and eval_env objects')
        
        if not self.fresh:
            success_prob = self._test_student(self.eval_env)
            self.trajectory.append((self.sched_idx, success_prob))
            print('SCHED_IDX', self.sched_idx)
            self.trans = self._interleave(self.training_env.get_attr('history'))
            self.trans_avg = self._average(self.training_env.get_attr('history'))
            print('AVG', self.trans_avg)
            print('TRAIN_SUC', np.mean(np.hstack(self.training_env.get_attr('history'))))
            print('TEST_SUC', success_prob)
            self.history[self.sched_idx].extend(self.trans)
            if self.logger:
                self.logger.record('trajectory/sched_idx', self.sched_idx)
                self.logger.record('trajectory/success_prob', success_prob)
                self.logger.record('trajectory/average', np.mean(self.trans))
            self._update_sched_idx()
        else:
            self.fresh = False   # not fresh after first iteration
        
        return {
            'iters': self.n_iters_per_ckpt,
            'map': self.trail_class(**self.sched(self.sched_idx))
        }

    def _interleave(self, histories):
        all_hist = [h for h in chain.from_iterable(zip_longest(*histories)) if h != None]
        return all_hist
    
    def _average(self, histories):
        avgs = []
        for hist in zip_longest(*histories):
            hs = [h for h in hist if h != None]

            # truncate outlier averages
            if len(hs) < len(hist) // 2: 
                print('TRUNC', hs)
                break

            avgs.append(np.mean(hs))
        
        return avgs
    
    def clear_hist(self, sched_idx):
        del self.history[sched_idx]

    def _update_sched_idx(self):
        raise NotImplementedError('implement _update_sched_idx() in child class')

    def _test_student(self, env):
        total_success = 0
        
        for _ in range(self.n_test_episodes):
            obs = env.reset()
            is_done = np.array(self.eval_env.num_envs * [False])
            is_success = np.zeros(self.eval_env.num_envs)

            while not np.all(is_done):
                # print('IS_DONE', is_done)
                action, _ = self.student.predict(obs, deterministic=True)
                # print('ACTION', action)
                obs, _, is_done_curr, info = env.step(action)
                # print('INFO', info)
                is_success_curr = np.array([i['is_success'] for i in info])

                is_success[~is_done] = is_success_curr[~is_done]
                is_done[~is_done] = is_done_curr[~is_done]

                # print('IS_SUC', is_success)

            total_success += np.sum(is_success)
        
        # print('TOT_SUC', total_success)
        return total_success / (self.n_test_episodes * len(is_done))



class FinalTaskTeacher(Teacher):
    def __init__(self, tau=0.95, **teacher_kwargs):
        super().__init__(**teacher_kwargs)
        self.prob_threshold = tau
        self.sched_idx = self.sched_len - 1
    
    def _update_sched_idx(self):
        assert self.sched_idx == self.sched_len - 1

        _, prob = self.trajectory[-1]
        if prob >= self.prob_threshold:
            raise StopIteration

def env_fn(): return TrailEnv()

class IncrementalTeacher(Teacher):
    def __init__(self, tau=0.95, use_avg=True, discount=0.8, decision_point=None, aggressive_checking=False, **teacher_kwargs):
        super().__init__(**teacher_kwargs)
        self.discount = discount
        self.prob_threshold = tau
        self.decision_point = tau if decision_point == None else decision_point
        self.use_avg = use_avg
        self.aggressive_checking = aggressive_checking
        self.avgs = []

    def load_student(self, *args, **kwargs):
        super().load_student(*args, **kwargs)

        if self.aggressive_checking:
            self.target_env = SubprocVecEnv([env_fn for _ in range(self.eval_env.num_envs)])
            self.target_env.env_method('queue_map', self.trail_class(**self.sched(self.sched_len - 1)))
    
    def _update_sched_idx(self):
        _, prob = self.trajectory[-1]

        if self.aggressive_checking:
            if self.sched_idx < self.sched_len - 1:
                prob = self._test_student(self.target_env)

            if prob >= self.prob_threshold:
                raise StopIteration
        else:
            if self.sched_idx == self.sched_len - 1 and prob >= self.prob_threshold:
                raise StopIteration
        
        trans = self.trans
        if self.use_avg:
            trans = self.trans_avg

        self._consume_trans(trans)

        if self.avgs[-1] > self.decision_point:
            self.sched_idx = min(self.sched_idx + 1, self.sched_len - 1)

    def _consume_trans(self, trans):
        avg = self.avgs[-1] if len(self.avgs) > 0 else 0
        for x in trans:
            avg = (1 - self.discount) * x + self.discount * avg
        self.avgs.append(avg)
        self.logger.record('trajectory/exp_avg', avg)


class AdaptiveOscTeacher(Teacher):
    def __init__(self, tau=0.95, conf=0.2, min_m_abs=5, max_m_factor=3, **teacher_kwargs):
        super().__init__(**teacher_kwargs)
        self.tau = tau
        self.conf = conf

        raw_min_m = np.log(1 - conf) / np.log(tau) - 1
        self.min_m = max(int(np.floor(raw_min_m)), min_m_abs)
        self.max_m = int(self.min_m * max_m_factor)
        self.curr_idx = 0
    
    def _update_sched_idx(self):
        trans = self.history[self.sched_idx]
        _, prob = self.trajectory[-1]

        if self.curr_idx == self.sched_len - 1 and prob >= self.tau:
            raise StopIteration

        if self.do_jump(trans):
            self.curr_idx = min(self.curr_idx + 1, self.sched_len - 1)
            self.sched_idx = self.curr_idx
        elif self.do_dive(trans):
            self.curr_idx = max(self.curr_idx - 1, 0)
            self.sched_idx = self.curr_idx
        else:
            if self.sched_idx == self.curr_idx:
                self.sched_idx = max(self.curr_idx - 1, 0)
            else:
                self.sched_idx = self.curr_idx
        

    def do_jump(self, trans):
        for k in range(self.min_m, 1 + min(self.max_m, len(trans))):
            prob_good = self._get_prob_good(trans[-k:])
            if prob_good >= self.conf:
                return True

        return False

    def do_dive(self, trans):
        rev_trans = [not bit for bit in trans]
        for k in range(self.min_m, 1 + min(self.max_m, len(trans))):
            prob_good = self._get_prob_good(rev_trans[-k:])
            if prob_good >= self.conf:
                return True
        
        return False

    def _get_prob_good(self, transcript):
        success = np.sum(transcript)
        total = len(transcript)
        prob_bad = beta.cdf(self.tau, a=success+1, b=total-success+1)
        return 1 - prob_bad


class AdaptiveExpTeacher(Teacher):
    def __init__(self, tau=0.95, discount=0.8, decision_point=0.7, noise_range=0.1, use_avg=True, aggressive_checking=False, **teacher_kwargs):
        super().__init__(**teacher_kwargs)
        self.tau = tau
        self.discount = discount
        self.use_avg = use_avg
        self.decision_point = decision_point
        self.noise_range = noise_range
        self.avgs = []
        self.aggressive_checking = aggressive_checking

    def load_student(self, *args, **kwargs):
        super().load_student(*args, **kwargs)

        if self.aggressive_checking:
            self.target_env = SubprocVecEnv([env_fn for _ in range(self.eval_env.num_envs)])
            self.target_env.env_method('queue_map', self.trail_class(**self.sched(self.sched_len - 1)))

    def _update_sched_idx(self):
        _, prob = self.trajectory[-1]

        trans = self.trans
        if self.use_avg:
            trans = self.trans_avg

        self._consume_trans(trans)

        if self.aggressive_checking:
            if self.sched_idx < self.sched_len - 1:
                prob = self._test_student(self.target_env)

            if prob >= self.tau:
                raise StopIteration
        else:
            if self.sched_idx == self.sched_len - 1 and prob >= self.tau:
                raise StopIteration

        if len(self.avgs) == 1:
            return
        
        avg, last_avg = self.avgs[-1], self.avgs[-2]

        if avg > self.decision_point + self.noise_range:
            if avg >= last_avg:
                self.sched_idx = min(self.sched_idx + 1, self.sched_len - 1)
            else:
                # self.sched_idx = max(self.sched_idx - 1, 0)
                return
        if avg < self.decision_point - self.noise_range:
            if avg < last_avg:
                self.sched_idx = max(self.sched_idx - 1, 0)

    def _consume_trans(self, trans):
        avg = self.avgs[-1] if len(self.avgs) > 0 else 0
        for x in trans:
            avg = (1 - self.discount) * x + self.discount * avg
        self.avgs.append(avg)
        self.logger.record('trajectory/exp_avg', avg)


class AdaptiveDoubleExpTeacher(Teacher):

    def __init__(self, tau=0.95, data_discount=0.8, trend_discount=0.9, **teacher_kwargs):
        super().__init__(**teacher_kwargs)
        self.tau = tau
        self.data_discount = data_discount
        self.trend_discount = trend_discount

        self.data_hist = []
        self.trend_hist = []

    def _update_sched_idx(self):
        _, prob = self.trajectory[-1]
        self._consume_trans(self.trans)

        data_avg = self.data_hist[-1]
        trend_avg = self.trend_hist[-1]

        if data_avg > 0.7:
            if trend_avg > 0:
                self.sched_idx = min(self.sched_idx + 1, self.sched_len - 1)
            else:
                # self.sched_idx = max(self.sched_idx - 1, 0)
                return
        else:
            if trend_avg > 0:
                return
            else:
                self.sched_idx = max(self.sched_idx - 1, 0)

        if self.sched_idx == self.sched_len - 1 and prob >= self.tau:
            raise StopIteration

    def _consume_trans(self, trans):
        if len(self.data_hist) == 0:
            self.data_hist.append(0)
            self.trend_hist.append(0)
        
        last_data_avg = self.data_hist[-1]
        trend_avg = self.trend_hist[-1]

        for x in trans:
            data_avg = (1 - self.data_discount) * x + self.data_discount * (last_data_avg + trend_avg)
            trend_avg = (1 - self.trend_discount) * (data_avg - last_data_avg) + self.trend_discount * trend_avg
            last_data_avg = data_avg

        self.data_hist.append(last_data_avg)
        self.trend_hist.append(trend_avg)

        self.logger.record('trajectory/data_avg', last_data_avg)
        self.logger.record('trajectory/trend_avg', trend_avg)


class AdaptiveOscTeacherCont(Teacher):
    def __init__(self, goal_length, tau=0.5, threshold=0.8, threshold_low=0.2, conf=0.95, cut_factor=2, min_m_abs=5, max_m_factor=3, **teacher_kwargs):
        super().__init__(**teacher_kwargs)
        self.goal_length = goal_length
        self.tau = tau
        self.threshold = threshold
        self.threshold_low = threshold_low
        self.cut_factor = cut_factor
        self.conf = conf

        raw_min_m = np.log(1 - conf) / np.log(threshold) - 1
        self.min_m = max(int(np.floor(raw_min_m)), min_m_abs)
        self.max_m = int(self.min_m * max_m_factor)
        self.mid_m = (self.min_m + self.max_m) // 2

        self.sched_idx = goal_length // cut_factor
        self.inc = None

    # TODO: always clear history?
    def _update_sched_idx(self):
        trans = self.history[self.sched_idx]
        _, prob = self.trajectory[-1]

        if self.inc != None:
            if len(trans) > self.min_m:
                if self.do_jump(trans):
                    self.clear_hist(self.sched_idx)
                    self.sched_idx = min(self.sched_idx + self.inc, self.goal_length)
                elif self.do_dive(trans):
                    self.clear_hist(self.sched_idx)
                    self.sched_idx //= self.cut_factor
                    self.inc //= self.cut_factor
        elif len(trans) >= self.max_m:
            if self.do_jump(trans, thresh=self.tau):
                self.inc = self.sched_idx
            else:
                self.clear_hist(self.sched_idx)
                self.sched_idx = max(self.sched_idx // self.cut_factor, 1)

        if self.sched_idx == self.goal_length and prob >= 0.95:  # TODO: hardcoded
            raise StopIteration

    def do_jump(self, trans, thresh=None):
        print('ESTM', np.mean(trans))
        print('THRESH', thresh)
        print('TRANS', trans[-self.max_m:])
        print('TRANS ALL', trans)
        for k in range(self.min_m, 1 + min(self.max_m, len(trans))):
            prob_good = self._get_prob_good(trans[-k:], thresh=thresh)
            print("PROB", prob_good)
            if prob_good >= self.conf:
                return True

        return False

    def do_dive(self, trans):
        rev_trans = [not bit for bit in trans]
        return self.do_jump(rev_trans, 1 - self.threshold_low)

    def _get_prob_good(self, transcript, thresh=None):
        if thresh == None:
            thresh = self.threshold

        success = np.sum(transcript)
        total = len(transcript)
        prob_bad = beta.cdf(thresh, a=success+1, b=total-success+1)
        return 1 - prob_bad
        

class RandomTeacher(Teacher):
    def __init__(self, tau=0.95, **teacher_kwargs):
        super().__init__(**teacher_kwargs)
        self.prob_threshold = tau
        self.sched_idx = np.random.choice(self.sched_len)

    def load_student(self, *args, **kwargs):
        super().load_student(*args, **kwargs)
        self.target_env = SubprocVecEnv([env_fn for _ in range(self.eval_env.num_envs)])
        self.target_env.env_method('queue_map', self.trail_class(**self.sched(self.sched_len - 1)))
    
    def _update_sched_idx(self):
        _, prob = self.trajectory[-1]
        if self.sched_idx < self.sched_len - 1:
            prob = self._test_student(self.target_env)

        if prob >= self.prob_threshold:
            raise StopIteration
        
        self.sched_idx = np.random.choice(self.sched_len)
    
    