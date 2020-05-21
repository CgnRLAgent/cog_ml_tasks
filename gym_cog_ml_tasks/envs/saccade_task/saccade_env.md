## SACCADE
The Saccade-Antisaccade task consists in a sequence of multi-step trials where the final goal is to direct the eye movement according to previous cues projected on a screen.  
The cues are essentially of two types:  
- fixation mark : square at the center of the screen; if it is black it is a pro-saccade trial (P), if it is white it is an anti-saccade one (A);

- location cue: circle at the side of the screen; it can be either at the left side (L) or at the right side (R) of the screen
The test-taker has 3 possible activities: F=front,L=left, R=right.
In case of a pro-saccade trial, the eye movement has to be in the same direction as the location cue (PL or PR); otherwise, it has to be in the opposite direction (AL or AR).

Each trial is composed by different phases:
- START: empty screen
- FIX: fixation mark appears; the test-taker has to fix it for two consecutive timesteps to have a first reward r_f (F selected twice)
- CUE: location cue appears together with fixation mark
- DELAY: location cue disappears for two timesteps to test the memory delay
- GO: fixation mark disappears as well (empty screen) and the subject has to solve the task (it has up to 8 timesteps to answer L or R)

**actions:**  'F', 'L', 'R'

**predefined rewards:** 
- FIX: 1.0 * 2 timestep for correct action, 0 * 2 for wrong; 
- CUE: 0 for correct action, -1 for wrong;
- DELAY: 0 for correct action, -1 for wrong;
- GO: 10 for correct action, 0 for wrong.

e.g.

Start:
Time 1-2, Screen(Fix mark): A; 
Target action: F; 
Reward: 1*2;

Time 3, Screen(Location cue): AL; 
Target action: F; 
Reward: 0/-1(if wrong)

Time 4-5, Screen(Delay): A; 
Target action: F; 
Reward: 0/-1(if wrong) * 2

Time 5, Screen(Go): 'empty'; 
Target action: R; 
Reward: 10

### Usage
```python
import gym
import gym_cog_ml_tasks

env = gym.make('Saccade-v0')
env.reset()
env.step(env.action_space.sample())
env.render()

# # custom params
# env = gym.make('Saccade-v0', go_reward=5)
```