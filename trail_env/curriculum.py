"""
Training schedules

author: William Tong (wtong@g.harvard.edu)
"""

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from trail_map import MeanderTrail

class CurriculumCallback(BaseCallback):
    def __init__(self, teacher, eval_env = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.teacher = teacher

        self.curr_iter = 0
        self.curr_gen = 0
    
    def _on_training_start(self) -> None:
        self.teacher.load_logger(self.logger)
        self.teacher.load_student(self.model, self.eval_env)
        self.curr_ckpt = self.teacher.next_checkpoint()
        self.training_env.env_method('queue_map', self.curr_ckpt['map'])

        if self.eval_env:
            self.eval_env.queue_map(self.curr_ckpt['map'])
    
    def _on_step(self) -> bool:
        self.curr_iter += 1
        if self.curr_iter > self.curr_ckpt['iters']:
            try:
                self.curr_ckpt = self.teacher.next_checkpoint()
            except StopIteration:
                return False

            self.training_env.env_method('queue_map', self.curr_ckpt['map'])
            self.curr_iter = 0

            if self.eval_env:
                self.eval_env.queue_map(self.curr_ckpt['map'])

            # self.model.save(f'gen{self.curr_gen}')
            # self.curr_gen += 1

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
    def __init__(self, len_sched=None):
        self.trail_class = MeanderTrail
        self.trail_args = {
            'width': 5,
            'diff_rate': 0.01,
            'radius': 100,
            'reward_dist': -1,
            'range': (-np.pi / 3, np.pi / 3)
        }

        self.length_schedule = len_sched if len_sched != None else [10, 20, 30]
        self.sched_idx = 0

        self.n_iters_per_ckpt = 1000
        self.n_test_episodes = 25
        self.student = None
        self.eval_env = None
        self.fresh = True
        self.trajectory = []
        self.logger = None
    
    def load_logger(self, logger):
        self.logger = logger

    def load_student(self, student, eval_env):
        self.student = student
        self.eval_env = eval_env
        self.fresh = True
        self.trajectory = []
    
    def next_checkpoint(self):
        if self.student == None or self.eval_env == None:
            raise Exception('student or eval_env not initialized: load_student() with student and eval_env objects')
        
        if not self.fresh:
            success_prob = self._test_student(self.student, self.eval_env)
            self.trajectory.append((self.sched_idx, success_prob))
            if self.logger:
                self.logger.record('trajectory/sched_idx', self.sched_idx)
                self.logger.record('trajectory/success_prob', success_prob)
            self._update_sched_idx()
        else:
            self.fresh = False   # not fresh after first iteration
        
        return {
            'iters': self.n_iters_per_ckpt,
            'map': self.trail_class(length=self.length_schedule[self.sched_idx], **self.trail_args)
        }

    def _update_sched_idx(self):
        raise NotImplementedError('implement _update_sched_idx() in child class')

    def _test_student(self, student, env):
        total_success = 0

        for _ in range(self.n_test_episodes):
            is_done = False
            obs = env.reset()
            while not is_done:
                action, _ = student.predict(obs, deterministic=True)
                obs, _, is_done, info = env.step(action)
                is_success = info['is_success']
            
            if is_success:
                total_success += 1
        
        return total_success / self.n_test_episodes


class IncrementalTeacher(Teacher):
    def __init__(self, **teacher_kwargs):
        super().__init__(**teacher_kwargs)
        self.prob_threshold = 0.9
    
    def _update_sched_idx(self):
        _, prob = self.trajectory[-1]

        if prob > self.prob_threshold:
            self.sched_idx += 1
            if self.sched_idx == len(self.length_schedule):
                print('FINAL SCHED IDX', self.sched_idx)
                raise StopIteration


class RandomTeacher(Teacher):
    def __init__(self, env_class, target_env=None, **teacher_kwargs):  # TODO: env_class is hacky -- fix imports
        super().__init__(**teacher_kwargs)
        self.prob_threshold = 0.9

        if target_env == None:
            target_env = env_class(MeanderTrail(length=self.length_schedule[-1], **self.trail_args))
        self.target_env = target_env
    
    def _update_sched_idx(self):
        _, prob = self.trajectory[-1]
        if prob > self.prob_threshold:
            target_prob = self._test_student(self.student, self.target_env)
            if target_prob > self.prob_threshold:
                raise StopIteration
        
        self.sched_idx = np.random.choice(len(self.length_schedule))
    
    