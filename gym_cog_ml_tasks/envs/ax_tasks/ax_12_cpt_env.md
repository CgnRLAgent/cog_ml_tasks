## AX CPT TASK
The 12 AX CPT task consists in the presentation to the subject of 8 possible stimuli/cues: 3 context cues 'A' - 'B' - 'C' and 3 target cues 'X' - 'Y' - 'Z'.
The tester has 2 possible responses which depend on the temporal order of previous and current stimuli: 
he has to answer 'R' 
    when
- the last stored digit is '1' AND the current stimulus is 'X' AND the previous stimulus is 'A' , 
- the last stored digit is '2' AND the current stimulus is 'Y' AND the previous stimulus is 'B' , 
in any other case , reply 'L'.

**actions:**  'L', 'R'

**predefined rewards:** output correctly(1.0), not correct(-1.0)

e.g.

Input : 2AXBYAXCX1CYAZAXBYAX

Target: LLLLRLLLLLLLLLLRLLLR

Input:  1BYAXAXAXAXCXAXBYAXCX

Target: LLLLRLRLRLRLLLRLLLRLL

### Usage
```python
import gym
import gym_cog_ml_tasks

env = gym.make('12AX_CPT-v0')
env.reset()
env.step(env.action_space.sample())
env.render()
# # custom params
# env = gym.make('12AX_CPT-v0', size=800, prob_target=0.5)
# Due to the randomness, the actual size could be either 800 or 801. Default size is 1000.
```