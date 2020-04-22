## AX_CPT
The AX_CPT task consists in the presentation to the subject of four possible stimuli/cues: two context cues 'A' - 'B' and 2 target cues 'X' - 'Y'.

The tester has 2 possible responses which depend on the temporal order of previous and current stimuli: 
he has to answer 'R' when
* the current stimulus is 'X' AND the previous stimulus is 'A' ,
in any other case , reply 'L'.

**actions:**  'L', 'R'

**predefined rewards:** output correctly(1.0), not correct(-1.0)

e.g.

Input: AXBY

Target: LRLL

Input: XXYAX

Target: LLLLR

### Usage
```python
import gym
import gym_cog_ml_tasks

env = gym.make('AX_CPT-v0')
env.reset()
env.step(env.action_space.sample())
env.render()
# custom params
env = gym.make('AX_CPT-v0', size=500)
```