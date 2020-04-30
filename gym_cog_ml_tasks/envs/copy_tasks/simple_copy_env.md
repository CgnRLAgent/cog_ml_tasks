## Simpe_Copy
Simply copy the input sequence as output.

At each time step a character is observed, and the agent should respond a char.
The action(output) is chosen from a char set e.g. {A,B,C,D,E}.

If the agent respond an incorrect char, the episode will end.

**predefined rewards:** output correctly(1.0), not correct(-1.0)

e.g.

Input:   ABCDE

Target:  ABCDE

### Usage
```python
import gym
import gym_cog_ml_tasks

env = gym.make('Simple_Copy-v0')
env.reset()
env.step(env.action_space.sample())
env.render()
# custom params
env = gym.make('Simple_Copy-v0', n_char=3, len_range=(5,10))
```