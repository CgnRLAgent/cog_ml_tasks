"""
AX CPT TASK:

he AX CPT task consists in the presentation to the subject of four possible stimuli/cues: two context cues 'A' - 'B' and 2 target cues 'X' - 'Y'.
This task must start with context cues. Context cues and target cues take turns to appear. 
The tester has 2 possible responses which depend on the temporal order of previous and current stimuli: 
he has to answer 'R' when
- the current stimulus is 'X' AND the previous stimulus is 'A' ,
in any other case , reply 'L'.

AUTHOR: dcyril233
DATE: 04.2020
"""

from gym import Env
from gym.spaces import Discrete
from gym.utils import colorize, seeding
import numpy as np
import sys


class AX_CPT_ENV(Env):

    CHAR_1 = ['A', 'B']
    CHAR_2 = ['X', 'Y']
    ACTIONS = ['L', 'R']

    def __init__(self, size=500):
        """
        :param size: the number of inputing stimuli/cues
        """
        # observation (characters)
        self.idx_2_char = self.CHAR_1 + self.CHAR_2
        self.char_2_idx = {}
        for i, c in enumerate(self.idx_2_char):
            self.char_2_idx[c] = i
        self.observation_space = Discrete(len(self.idx_2_char))

        # action
        self.action_space = Discrete(len(self.ACTIONS))

        self.size = size

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
        assert 0 <= self.position <= self.input_length
        output = self.ACTIONS[action]
        reward = 1.0 if output == self.target_str[self.position] else -1.0
        self.last_action = action
        self.last_reward = reward
        self.episode_total_reward += reward
        self.output_str += output
        self.position += 1
        if self.position < self.input_length:
            done = False
            _, obs = self._get_observation()
        else:
            done = True
            obs = None
        return obs, reward, done, {}

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
        for i in np.arange(size):
            if i % 2 == 0:
                s = np.random.choice(self.CHAR_1)
            else:
                s = np.random.choice(self.CHAR_2)
            input_str += s
            if len(input_str) > 1 and input_str[-2:] == 'AX':
                target_str += 'R' 
            else:
                target_str += 'L'
        return input_str, target_str

    def _get_observation(self, pos=None):
        if pos is None:
            pos = self.position
        obs_char = self.input_str[pos]
        obs_idx = self.char_2_idx[obs_char]
        return obs_char, obs_idx
