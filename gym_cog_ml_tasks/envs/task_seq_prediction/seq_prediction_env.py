"""
seq_prediction TASK:

Consider two abstract sequences A-B-C-D and X-B-C-Y
In this example remembering that the sequence started with A or X is required 
to make the correct prediction following C. 

AUTHOR: JiqingFeng
DATE: 05.2020
"""

from gym import Env
from gym.spaces import Discrete
from gym.utils import colorize, seeding
import numpy as np
import sys


class seq_prediction_ENV(Env):

    STR_in = ['ABC', 'XBC']
    CHAR_in = ['A', 'B', 'C', 'X']
    ACTIONS = ['B', 'C', 'D', 'Y']

    def __init__(self, size=100, p=0.5):
        """
        :param size: the number of inputing stimuli/cues
        :param p: the probability to generate 'ABC' or 'XBC'
        """
        # observation (characters)
        self.idx_2_char = self.CHAR_in
        self.char_2_idx = {}
        for i, c in enumerate(self.idx_2_char):
            self.char_2_idx[c] = i
        self.observation_space = Discrete(len(self.idx_2_char))

        # action
        self.action_space = Discrete(len(self.ACTIONS))

        self.size = size
        self.p = p
        
        # states of an episode
        self.position = None
        self.last_action = None
        self.last_reward = None
        self.episode_total_reward = None
        self.input_str = None
        self.target_str = None
        self.output_str = None

        self.np_random = None
        self.seed()
        self.reset()

    @property
    def input_length(self):
        return len(self.input_str)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.position = 0
        self.last_action = None
        self.last_reward = None
        self.episode_total_reward = 0.0
        self.input_str, self.target_str = self._generate_input_target(self.size)
        self.output_str = ''
        obs_char, obs_idx = self._get_observation()
        return obs_idx

    def step(self, action):
        assert self.action_space.contains(action)
        assert 0 <= self.position < self.input_length
        target_act = self.ACTIONS.index(self.target_str[self.position])
        reward = 1.0 if action == target_act else -1.0
        self.last_action = action
        self.last_reward = reward
        self.episode_total_reward += reward
        self.output_str += self.ACTIONS[action]
        self.position += 1
        if self.position < self.input_length:
            done = False
            _, obs = self._get_observation()
        else:
            done = True
            obs = None
        info = {"target_act": target_act}
        return obs, reward, done, info

    def render(self, mode='human'):
        outfile = sys.stdout  #TODO: other mode
        pos = self.position - 1
        if pos > -1:
            o_str = self.output_str[:pos]
            color = 'green' if self.target_str[pos] == self.output_str[pos] else 'red'
            o_str += colorize(self.output_str[pos], color, highlight=True)
        else:
            o_str = ''
        outfile.write("="*20 + "\n")
        outfile.write("Length   : " + str(self.input_length) + "\n")
        outfile.write("Input    : " + self.input_str + "\n")
        outfile.write("Target   : " + self.target_str + "\n")
        outfile.write("Output   : " + o_str + "\n")
        if self.position > 0:
            outfile.write("-" * 20 + "\n")
            outfile.write("Current reward:   %.2f\n" % self.last_reward)
            outfile.write("Cumulative reward:   %.2f\n" % self.episode_total_reward)
        outfile.write("\n")
        return

    def _generate_input_target(self, size):
        input_str = ''
        target_str = ''
        for _ in np.arange(int(size/3)):
            s = np.random.choice(self.STR_in, p=[self.p, 1-self.p])
            input_str += s
            if s == 'ABC':
                target_str += 'BCD'
            else:
                target_str += 'BCY'
        remainder = int(size % 3)
        input_str += np.random.choice(self.STR_in, p=[self.p, 1-self.p])[:remainder]
        if remainder == 1:
            target_str += 'B'
        elif remainder == 2:
            target_str += 'BC'
        return input_str, target_str

    def _get_observation(self, pos=None):
        if pos is None:
            pos = self.position
        obs_char = self.input_str[pos]
        obs_idx = self.char_2_idx[obs_char]
        return obs_char, obs_idx