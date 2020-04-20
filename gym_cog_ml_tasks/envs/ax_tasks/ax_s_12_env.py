"""
AX-S_12 TASK:

The AX_12 task consists in the presentation to the subject of six possible stimuli/cues '1' - '2', 'A' - 'B', 'X' - 'Y'.
In AX-S 12 task, there are extra cues like '1+','2+','A+','B+','C+','X+','Y+','Z+','1-','2-','A-','B-','C-','X-','Y-','Z-'.
Each cue comes with a sign indicate postive or negative e.g: ('A+' or 'A-')

The tester has 2 possible responses which depend on the temporal order of previous and current stimuli:
he has to answer 'R' when
- the last stored digit is '1' AND the previous stimulus is 'A' AND the current one is 'X',
- the last stored digit is '2' AND the previous stimulus is 'B' AND the current one is 'Y';
in any other case , reply 'L'.

AUTHOR: Xingchen
DATE: 04.2020
"""

from gym import Env
from gym.spaces import Discrete
from gym.utils import colorize, seeding
import numpy as np
import sys
import re

class AX_S_12_ENV(Env):

    DIGITS = ['1', '2']
    CHAR_1 = ['A', 'B', 'C']
    CHAR_2 = ['X', 'Y', 'Z']

    DIGITS_S = ['1-', '2-', '1+' ,'2+']
    CHAR_1_S = ['A+','B+','C+','A-','B-','C-']
    CHAR_2_S = ['X+','Y+','Z+','X-','Y-','Z-']
    ACTIONS = ['L', 'R']
    # SIGN = ['+','-']


    def __init__(self, min_size=1, prob_r=0.3):
        """
        :param min_size: the min number of sets of 3-char combinations of generated inputs, e.g. 1: 1AX; 2: 1AXBY; 3: 1AXBYCZ
        :param prob_r: the probability to generate 'AX' given '1' or 'BY' given '2'
        """


        #observation
        self.idx_2_char_s = self.DIGITS_S + self.CHAR_1_S + self.CHAR_2_S
        self.idx_2_char = self.DIGITS + self.CHAR_1 + self.CHAR_2
        self.char_2_idx = {}

        for i, c in enumerate((self.idx_2_char+self.idx_2_char_s+self.char_sets)):
                self.char_2_idx[c] = i
        self.observation_space = Discrete(len(self.idx_2_char))



        # action
        self.action_space = Discrete(len(self.ACTIONS))

        self.min_size = min_size
        self.prob_r = prob_r


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
    def char_sets(self):
        sets = []
        for c1 in (self.CHAR_1 + self.CHAR_1_S):
            for c2 in (self.CHAR_2 + self.CHAR_2_S):
                sets.append(c1 + c2)

        return sets


    @property
    def probs_given_1(self):
        n_sets = len(self.char_sets)
        prob_l = (1 - self.prob_r) / (n_sets - 1)
        p = np.full(n_sets, prob_l)
        p[self.char_sets.index('AX')] = self.prob_r
        return p

    
    @property
    def probs_given_2(self):
        n_sets = len(self.char_sets)
        prob_l = (1 - self.prob_r) / (n_sets - 1)
        p = np.full(n_sets, prob_l)
        p[self.char_sets.index('BY')] = self.prob_r
        return p


    @property
    def input_length(self):

        str_ = self.listToString(self.input_str)
        result = ''
        for s in str_:
            if s != '+' and s != '-':
                result += s
        return len(result)

    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]    

        def reset(self):
            self.position = 0
        self.last_action = None
        self.last_reward = None
        self.episode_total_reward = 0.0
        size = self.np_random.randint(3) + self.min_size
        self.input_str, self.target_str = self._generate_input_target(size)
        self.output_str = ''
        obs_char, obs_idx = self._get_observation()
        return obs_idx

    def reset(self):
        self.position = 0
        self.last_action = None
        self.last_reward = None
        self.episode_total_reward = 0.0
        size = self.np_random.randint(3) + self.min_size
        self.input_str, self.target_str = self._generate_input_target(size)
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


    def listToString(self, s):  
    
        # initialize an empty string 
        str1 = ""  
        
        # traverse in the string   
        for ele in s:  
            str1 += ele   
        
        # return string   
        return str1

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
        outfile.write("Input    : " + self.listToString(self.input_str) + "\n")
        outfile.write("Target   : " + self.listToString(self.target_str) + "\n")
        outfile.write("Output   : " + o_str + "\n")
        if self.position > 0:
            outfile.write("-" * 20 + "\n")
            outfile.write("Current reward:   %.2f\n" % self.last_reward)
            outfile.write("Cumulative reward:   %.2f\n" % self.episode_total_reward)
        outfile.write("\n")
        return



    def _generate_input_target(self, size):
        input_str = []
        target_str = []

        digit = np.random.choice(self.DIGITS+self.DIGITS_S)
        input_str.append(digit)
        target_str.append('L')
        for _ in np.arange(size):
            if digit == '1':
                s = np.random.choice(self.char_sets, p=self.probs_given_1)
                # print('s',s)
                input_str.append(s)
                target_str.append('LR') if s == 'AX' else target_str.append('LL')
            else:
                s = np.random.choice(self.char_sets, p=self.probs_given_2)
                # print('s',s)
                input_str.append(s)
                target_str.append('LR') if s == 'BY' else target_str.append('LL')
        return input_str, target_str

    def _get_observation(self, pos=None):
        if pos is None:
            pos = self.position
        obs_char = self.input_str[pos]
        # print('obs_char:',obs_char)
        obs_idx = self.char_2_idx[obs_char]
        return obs_char, obs_idx        