"""
Simple binary environments for experimenting with teacher-student interactions

author: William Tong (wtong@g.harvard.edu)
"""

import gym

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


# TODO: flesh out curriculum dynamics
class CurriculumEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        self.observation_space = None
        self.action_space = None
    
    def step(self, action):
        return None, None, None, {}
    
    def reset(self):
        return None
