"""
Training schedules

author: William Tong (wtong@g.harvard.edu)
"""

from collections import defaultdict
from itertools import chain, zip_longest
from pathlib import Path
import numpy as np
from scipy.stats import beta

from stable_baselines3.common.callbacks import BaseCallback

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
            self.save_path.mkdir(parents=True)

        self.curr_iter = 0
        self.curr_gen = 0
    
    def _on_training_start(self) -> None:
        self.teacher.load_logger(self.logger)
        self.teacher.load_student(self.model, self.eval_env)
        self.teacher.load_training_env(self.training_env)
        self.curr_ckpt = self.teacher.next_checkpoint()
        self.training_env.env_method('queue_map', self.curr_ckpt['map'])

        if self.eval_env:
            self.eval_env.queue_map(self.curr_ckpt['map'])
    
    def _on_step(self) -> bool:
        self.curr_iter += 1
        if self.curr_iter > self.curr_ckpt['iters']:
            for cb in self.next_lesson_callbacks:
                cb(self)

            try:
                self.curr_ckpt = self.teacher.next_checkpoint()
            except StopIteration:
                return False

            self.training_env.env_method('queue_map', self.curr_ckpt['map'])
            self.curr_iter = 0

            if self.eval_env:
                self.eval_env.queue_map(self.curr_ckpt['map'])

            self.curr_gen += 1
            if self.save_every > 0 and self.curr_gen % self.save_every == 0:
                self.model.save(str(self.save_path / f'gen{self.curr_gen}'))

        return True


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
    def __init__(self, sched):
        self.trail_class = MeanderTrail

        self.sched = sched
        self.sched_idx = 0

        self.n_iters_per_ckpt = 2000
        self.n_test_episodes = 25
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
    
    def load_training_env(self, env):
        self.training_env = env
    
    def next_checkpoint(self):
        if self.student == None or self.eval_env == None:
            raise Exception('student or eval_env not initialized: load_student() with student and eval_env objects')
        
        if not self.fresh:
            success_prob = self._test_student(self.eval_env)
            self.trajectory.append((self.sched_idx, success_prob))
            self.history[self.sched_idx].extend(self._interleave(self.training_env.get_attr('history')))
            if self.logger:
                self.logger.record('trajectory/sched_idx', self.sched_idx)
                self.logger.record('trajectory/success_prob', success_prob)
            self._update_sched_idx()
        else:
            self.fresh = False   # not fresh after first iteration
        
        return {
            'iters': self.n_iters_per_ckpt,
            'map': self.trail_class(**self.sched[self.sched_idx])
        }

    def _interleave(self, histories):
        all_hist = [h for h in chain.from_iterable(zip_longest(*histories)) if h != None]
        return all_hist

    def _update_sched_idx(self):
        raise NotImplementedError('implement _update_sched_idx() in child class')

    def _test_student(self, env):
        total_success = 0

        for _ in range(self.n_test_episodes):
            is_done = False
            obs = env.reset()
            while not is_done:
                action, _ = self.student.predict(obs, deterministic=True)
                obs, _, is_done, info = env.step(action)
                is_success = info['is_success']
            
            if is_success:
                total_success += 1
        
        return total_success / self.n_test_episodes
    

class NaiveTeacher(Teacher):
    def __init__(self, tau=0.95, len_sched=None):
        super().__init__(len_sched)
        self.prob_threshold = tau
    
    def _update_sched_idx(self):
        _, prob = self.trajectory[-1]
        self.sched_idx = len(self.sched) - 1

        if prob > self.prob_threshold:
            raise StopIteration


class IncrementalTeacher(Teacher):
    def __init__(self, tau=0.95, **teacher_kwargs):
        super().__init__(**teacher_kwargs)
        self.prob_threshold = tau
    
    def _update_sched_idx(self):
        _, prob = self.trajectory[-1]

        if prob > self.prob_threshold:
            self.sched_idx += 1
            if self.sched_idx == len(self.sched):
                raise StopIteration


class OscillatingTeacher(Teacher):
    def __init__(self, tau=0.95, conf=0.2, max_m_factor=3, **teacher_kwargs):
        super().__init__(**teacher_kwargs)
        self.tau = tau
        self.conf = conf

        raw_min_m = np.log(1 - conf) / np.log(tau) - 1
        self.min_m = int(np.floor(raw_min_m))
        self.max_m = int(self.min_m * max_m_factor)
        self.curr_idx = 0
    
    def _update_sched_idx(self):
        trans = self.history[self.sched_idx]
        _, prob = self.trajectory[-1]

        if self.do_jump(trans):
            self.curr_idx = min(self.curr_idx + 1, len(self.sched) - 1)
            self.sched_idx = self.curr_idx
        elif self.do_dive(trans):
            self.curr_idx = max(self.curr_idx - 1, 0)
            self.sched_idx = self.curr_idx
        else:
            if self.sched_idx == self.curr_idx:
                self.sched_idx = max(self.curr_idx - 1, 0)
            else:
                self.sched_idx = self.curr_idx
        
        if self.curr_idx == len(self.sched) - 1 and prob > self.tau:
            raise StopIteration

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


class RandomTeacher(Teacher):
    def __init__(self, target_env=None, **teacher_kwargs):
        super().__init__(**teacher_kwargs)
        self.prob_threshold = 0.9

        if target_env == None:
            target_env = TrailEnv(MeanderTrail(**self.sched[self.sched_idx]))
        self.target_env = target_env
    
    def _update_sched_idx(self):
        _, prob = self.trajectory[-1]
        if prob > self.prob_threshold:
            target_prob = self._test_student(self.target_env)
            if target_prob > self.prob_threshold:
                raise StopIteration
        
        self.sched_idx = np.random.choice(len(self.length_schedule))
    
    