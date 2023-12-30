"""
Curriculum implementations for the tracking tasks.

author: William Tong (wtong@g.harvard.edu)
"""

from collections import defaultdict
from itertools import chain, zip_longest
from pathlib import Path

import numpy as np

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
        return super()._on_rollout_end()

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
            self.trans = self._interleave(self.training_env.get_attr('history'))
            self.trans_avg = self._average(self.training_env.get_attr('history'))
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
                action, _ = self.student.predict(obs, deterministic=True)
                obs, _, is_done_curr, info = env.step(action)
                is_success_curr = np.array([i['is_success'] for i in info])

                is_success[~is_done] = is_success_curr[~is_done]
                is_done[~is_done] = is_done_curr[~is_done]

            total_success += np.sum(is_success)
        
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


class AdaptiveTeacher(Teacher):
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

# TODO: remove before end
AdaptiveExpTeacher = AdaptiveTeacher

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
    
    