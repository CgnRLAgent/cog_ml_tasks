## seq_prediction

Consider two abstract sequences A-B-C-D and X-B-C-Y
In this example remembering that the sequence started with A or X is required 
to make the correct prediction following C. 

**actions:**  'B', 'C', 'D', 'Y'

**predefined rewards:** output correctly(1.0), not correct(-1.0)

e.g.

Input: ABCXBC

Target: BCDBCY

Input: XBCXBCAB

Target: BCYBCYBC

### Usage
```python
import gym
import gym_cog_ml_tasks

env = gym.make('seq_prediction-v0')
env.reset()
env.step(env.action_space.sample())
env.render()
# custom params
env = gym.make('seq_prediction-v0', size=100, p=0.5)
```