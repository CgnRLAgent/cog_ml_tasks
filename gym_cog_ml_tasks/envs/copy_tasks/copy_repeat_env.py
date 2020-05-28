"""
simple copy-repeat task:

Copy the input sequence multi-times and reverse it every other time as output. For example:
(repeat time: 3)
Input:         ABCDE
Ideal output:  ABCDEEBCDAABCDE

At each time step a character is observed, and the agent should respond a char.
The action(output) is chosen from a char set e.g. {A,B,C,D,E}.

After the last input char is observed, an empty symbol will be observed for each step before the episode is end.
The episode ends when the agent respond R*X times, where X is the input seq length and R is the repeat time.

AUTHOR: Zenggo
DATE: 04.2020
"""

from gym import Env
from gym.spaces import Discrete
from gym.utils import colorize, seeding
import numpy as np
import sys
import string


class Copy_Repeat_ENV(Env):

    ALPHABET = list(string.ascii_uppercase[:26])

    def __init__(self, n_char=5, size=6, repeat=3):
        """
        :param n_char: number of different chars in inputs, e.g. 3 => {A,B,C}
        :param size: the length of input sequence
        :param repeat: the expected repeat times of the target output
        """
        self.n_char = n_char
        self.size = size
        self.repeat = repeat

        # observation (characters)
        self.observation_space = Discrete(n_char+1)  # +1: empty symbol, whose index is n_char
        # action
        self.action_space = Discrete(n_char)

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

    @property
    def target_length(self):
        return self.input_length * self.repeat

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.position = 0
        self.last_action = None
        self.last_reward = None
        self.episode_total_reward = 0.0
        self.input_str, self.target_str = self._generate_input_target()
        self.output_str = ''
        obs_char, obs_idx = self._get_observation()
        return obs_idx

    def step(self, action):
        assert self.action_space.contains(action)
        assert 0 <= self.position < self.target_length
        target_act = self.ALPHABET.index(self.target_str[self.position])
        reward = 1.0 if action == target_act else -1.0
        self.last_action = action
        self.last_reward = reward
        self.episode_total_reward += reward
        self.output_str += self.ALPHABET[action]
        self.position += 1
        if self.position < self.target_length:
            done = False
            _, obs = self._get_observation()
        else:
            done = True
            obs = None
        info = {"target_act": target_act}
        return obs, reward, done, info

    def render(self, mode='human'):
        outfile = sys.stdout  # TODO: other mode
        pos = self.position - 1
        o_str = ""
        if pos > -1:
            for i, c in enumerate(self.output_str):
                color = 'green' if self.target_str[i] == c else 'red'
                o_str += colorize(c, color, highlight=True)
        outfile.write("=" * 20 + "\n")
        outfile.write("Length   : " + str(self.input_length) + "\n")
        outfile.write("T-Length : " + str(len(self.target_str)) + "\n")
        outfile.write("Input    : " + self.input_str + "\n")
        outfile.write("Target   : " + self.target_str + "\n")
        outfile.write("Output   : " + o_str + "\n")
        if self.position > 0:
            outfile.write("-" * 20 + "\n")
            outfile.write("Current reward:   %.2f\n" % self.last_reward)
            outfile.write("Cumulative reward:   %.2f\n" % self.episode_total_reward)
        outfile.write("\n")
        return

    def _generate_input_target(self):
        input_str = ""
        for i in range(self.size):
            c = self.np_random.choice(self.ALPHABET[:self.n_char])
            input_str += c
        target_str = ""
        for i in range(self.repeat):
            if i % 2 == 1:
                target_str += input_str[::-1]
            else:
                target_str += input_str
        return input_str, target_str

    def _get_observation(self, pos=None):
        if pos is None:
            pos = self.position
        if pos >= self.input_length:
            obs_char = ''
            obs_idx = self.n_char
        else:
            obs_char = self.input_str[pos]
            obs_idx = self.ALPHABET.index(obs_char)
        return obs_char, obs_idx