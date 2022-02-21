"""
Simple binary environments for experimenting with teacher-student interactions

author: William Tong (wtong@g.harvard.edu)
"""

import gym
import numpy as np

from agent import Student

class BinaryEnv(gym.Env):
    def __init__(self, length, reward=1) -> None:
        super().__init__()
        self.length = length

        self.observation_space = gym.spaces.Discrete(length + 1)
        self.action_space = gym.spaces.Discrete(2)

        self.reward = reward
        self.loc = 0
    
    def step(self, action):
        self.loc += action
        reward = 0
        is_done = False

        if self.loc == self.length:
            reward = self.reward
            is_done = True
            
        return self.loc, reward, is_done, {}
    
    def reset(self):
        self.loc = 0
        return 0


class CurriculumEnv(gym.Env):
    def __init__(self, student, goal_length, train_iter, 
                 p_eps=0.05, 
                 teacher_reward=1,
                 student_reward=1):
        super().__init__()

        self.student = student
        self.goal_length = goal_length
        self.train_iter = train_iter
        self.p_eps = p_eps
        self.teacher_reward = teacher_reward
        self.student_reward = student_reward

        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(goal_length), 
            gym.spaces.Box(low=0, high=1, shape=(1,))))
        self.N = goal_length

        self.action_space = gym.spaces.Discrete(3)
    
    def step(self, action):
        d_length = action - 1
        self.N = np.clip(self.N + d_length, 0, self.goal_length)
        prob = self._get_score(self.N)

        reward = 0
        is_done = False

        if self.N == self.goal_length and (1 - prob) < self.p_eps:
            reward = self.teacher_reward
            is_done = True
        
        prob = self._get_score(self.N)
        return (self.N, prob), reward, is_done, {}
    
    def reset(self):
        student_score = self._get_score(self.goal_length)
        self.N  = self.goal_length
        return (self.goal_length, student_score)

    def _get_score(self, length):
        self.student.learn(BinaryEnv(length, reward=self.student_reward), max_iters=self.train_iter)
        return self.student.score(length)
    
