## AX_S_12
The AX_S_12 task consists in the presentation to the subject of six possible stimuli/cues '1' - '2', 'A' - 'B', 'X' - 'Y'.

The tester has 2 possible responses which depend on the temporal order of previous and current stimuli:
he has to answer 'R' when
* the last stored digit is '1' AND the previous stimulus is 'A' AND the current one is 'X',
* the last stored digit is '2' AND the previous stimulus is 'B' AND the current one is 'Y';
in any other case , reply 'L'.

**actions:**  'L', 'R'

**predefined rewards:** output correctly(1.0), not correct(-1.0)

e.g.

Input: 1BYAXBX

Target: LLLLRLL

Input: 2BYBXBX

Target: LLRLLLL

### Usage
```python
import gym
import gym_cog_ml_tasks

env = gym.make('AX_S_12-v0')
env.reset()
env.step(env.action_space.sample())
env.render()
# # custom params
# env = gym.make('AX_S_12-v0', size_range=(2,5), prob_target=0.5)
```