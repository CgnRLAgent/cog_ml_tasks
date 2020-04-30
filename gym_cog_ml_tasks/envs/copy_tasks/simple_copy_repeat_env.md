## Simpe_Copy_Repeat

Copy the input sequence multi-times and reverse it every other time as output. For example:

(repeat time: 3)

Input:         ABCDE

Ideal output:  ABCDEEBCDAABCDE

At each time step a character is observed, and the agent should respond a char.
The action(output) is chosen from a char set e.g. {A,B,C,D,E}.

After the last input char is observed, an empty symbol will be observed for each step before the episode is end.

The episode ends when the agent respond R*X times, where X is the input seq length and R is the repeat time.

If the agent respond an incorrect char, the episode will also end.

**predefined rewards:** output correctly(1.0), not correct(-1.0)

### Usage
```python
import gym
import gym_cog_ml_tasks

env = gym.make('Simple_Copy_Repeat-v0')
env.reset()
env.step(env.action_space.sample())
env.render()
# custom params
env = gym.make('Simple_Copy_Repeat-v0', n_char=3, len_range=(2,5), repeat_range=(3,4))
```