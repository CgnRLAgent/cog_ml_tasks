"""
simple copy task:

Simply copy the input sequence as output. For example:
Input:         ABCDE
Ideal output:  ABCDE

At each time step a character is observed, and the agent should respond a char.
The action(output) is chosen from a char set e.g. {A,B,C,D,E}.

AUTHOR: Zenggo
DATE: 04.2020
"""

from gym import Env
from gym.spaces import Discrete
from gym.utils import colorize, seeding
import numpy as np
import sys
import string


class Simple_Copy_v1_ENV(Env):

    ALPHABET = list(string.ascii_uppercase[:26])

    def __init__(self, n_char=5, size=10):
        """
        :param n_char: number of different chars in inputs, e.g. 3 => {A,B,C}
        :param size: the length of input sequence
        """
        self.n_char = n_char
        self.size = size

        # observation (characters)
        self.observation_space = Discrete(n_char)
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

        # generating mode
        self.mode = 'full'  # full, major, minor

        self.np_random = None
        self.seed()
        self.reset()

    @property
    def input_length(self):
        return len(self.input_str)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def setMode(self, m):
        assert m in ('full', 'major', 'minor')
        self.mode = m

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
        assert 0 <= self.position < self.input_length
        target_act = self.ALPHABET.index(self.target_str[self.position])
        reward = 1.0 if action == target_act else -1.0
        self.last_action = action
        self.last_reward = reward
        self.episode_total_reward += reward
        self.output_str += self.ALPHABET[action]
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
        outfile = sys.stdout  # TODO: other mode
        pos = self.position - 1
        o_str = ""
        if pos > -1:
            for i, c in enumerate(self.output_str):
                color = 'green' if self.target_str[i] == c else 'red'
                o_str += colorize(c, color, highlight=True)
        outfile.write("=" * 20 + "\n")
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

    def _generate_input_target(self):
        if self.mode == "full":
            input_str = self._gen_full()
        elif self.mode == "major":
            input_str = self._gen_major()
        else:
            input_str = self._gen_minor()
        target_str = input_str
        return input_str, target_str

    def _gen_full(self):
        input_str = ""
        candidates = self.ALPHABET[:self.n_char]
        for i in range(self.size):
            c = self.np_random.choice(candidates)
            input_str += c
        return input_str

    def _gen_major(self):
        input_str = ""
        for i in range(self.size):
            candidates = self.ALPHABET[:self.n_char]
            candidates.remove(candidates[i % self.n_char])
            c = self.np_random.choice(candidates)
            input_str += c
        return input_str

    def _gen_minor(self):
        is_minor = False
        input_str = ""
        while not is_minor:
            input_str = self._gen_full()
            for i in range(len(input_str)):
                # there is at least one char that doesn't appear in the
                # same position in major generation
                if input_str[i] == self.ALPHABET[i % self.n_char]:
                    is_minor = True
                    break
        return input_str

    def _get_observation(self, pos=None):
        if pos is None:
            pos = self.position
        obs_char = self.input_str[pos]
        obs_idx = self.ALPHABET.index(obs_char)
        return obs_char, obs_idx
