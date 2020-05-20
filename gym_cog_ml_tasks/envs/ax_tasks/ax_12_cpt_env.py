"""

12 AX CPT TASK
The 12 AX CPT task consists in the presentation to the subject of 8 possible stimuli/cues: 3 context cues 'A' - 'B' - 'C' and 3 target cues 'X' - 'Y' - 'Z'.
The tester has 2 possible responses which depend on the temporal order of previous and current stimuli: 
he has to answer 'R' 
    when
- the last stored digit is '1' AND the current stimulus is 'X' AND the previous stimulus is 'A' , 
- the last stored digit is '2' AND the current stimulus is 'Y' AND the previous stimulus is 'B' , 


in any other case , reply 'L'.


AUTHOR: Xingchen
DATE: 04.2020
"""


from gym import Env
from gym.spaces import Discrete
from gym.utils import colorize, seeding
import numpy as np
import sys



class AX_12_CPT_ENV(Env):
    
    DIGITS = ['1', '2']
    CHAR_1 = ['A', 'B', 'C']
    CHAR_2 = ['X', 'Y', 'Z']
    ACTIONS = ['L', 'R']

    def __init__(self, size=1000, prob_target=0.5,prob_12=0.1):
        """
        :param size: the number of digits and characters in total. Since it's random generated, the actual size may be size +1 
        :param prob_target: the probability to generate 'AX' or 'BY'
        """
        # observation (characters)
        self.idx_2_char = self.DIGITS + self.CHAR_1 + self.CHAR_2
        self.idx_2_char_s = self.CHAR_1 + self.CHAR_2
        self.char_2_idx = {}
        for i, c in enumerate(self.idx_2_char):
            self.char_2_idx[c] = i
        self.observation_space = Discrete(len(self.idx_2_char))

        # action
        self.action_space = Discrete(len(self.ACTIONS))

        self.size = size
        self.prob_target = prob_target
        self.prob_12 = prob_12
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
        for c1 in self.CHAR_1:
            for c2 in self.CHAR_2:
                sets.append(c1 + c2)

        for digit in self.DIGITS:
            sets.append(digit)
        return sets

    @property
    def probs(self):
        n_sets = len(self.char_sets)
        prob_other = (1 - self.prob_target-self.prob_12) / (n_sets - 4)
        p = np.full(n_sets, prob_other)
        
        p[self.char_sets.index('AX')] = self.prob_target / 2
        p[self.char_sets.index('BY')] = self.prob_target / 2
        p[self.char_sets.index('1')] = self.prob_12 / 2
        p[self.char_sets.index('2')] = self.prob_12 / 2
        return p

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
        self.input_str, self.target_str = self._generate_input_target()
        self.output_str = ''
        obs_char, obs_idx = self._get_observation()
        return obs_idx

    def step(self, action):
        assert self.action_space.contains(action)
        assert 0 <= self.position <= self.input_length
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

    def _generate_input_target(self):
        digit = np.random.choice(self.DIGITS)
        input_str = digit
        target_str = 'L'
        last_num = digit

        while len(input_str) < self.size:

            s = np.random.choice(self.char_sets, p=self.probs)
            input_str += s

            if s == '1':
                last_num = '1'
            elif s == '2':
                last_num = '2'

            if last_num == '1':
                if s in ('1','2'):
                    target_str += 'L'
                else:
                    target_str += 'LR' if s == 'AX' else 'LL'

            else:
                if s in ('1','2'):
                    target_str += 'L'
                else:
                    target_str += 'LR' if s == 'BY' else 'LL'
                

        return input_str, target_str

    def _get_observation(self, pos=None):
        if pos is None:
            pos = self.position
        obs_char = self.input_str[pos]
        obs_idx = self.char_2_idx[obs_char]
        return obs_char, obs_idx