# Cognitive and ML Tasks
This repo includes several tasks/environments built on [openai-gym](https://gym.openai.com/) for training RL agents. 

# Install
```
$ git clone https://github.com/CgnRLAgent/cog_ml_tasks.git
$ cd cog_ml_tasks
$ pip install -e .
```
**Requirements:** see the [setup.py](https://github.com/CgnRLAgent/cog_ml_tasks/blob/master/setup.py).

# Envs/Tasks Descriptions
## AX_12
The AX_12 task consists in the presentation to the subject of six possible stimuli/cues '1' - '2', 'A' - 'B', 'X' - 'Y'.

The tester has 2 possible responses which depend on the temporal order of previous and current stimuli:
he has to answer 'R' when
* the last stored digit is '1' AND the previous stimulus is 'A' AND the current one is 'X',
* the last stored digit is '2' AND the previous stimulus is 'B' AND the current one is 'Y';
in any other case , reply 'L'.

**actions:**  'L', 'R'

**predefined rewards:** output correctly(1.0), not correct(-1.0)

e.g.

Input: 1AXBZ

Target: LLRLL

Input: 2CXCYBY

Target: LLLLLLR

### Usage
```python
import gym
import gym_cog_ml_tasks

env = gym.make('AX_12-v0')
env.reset()
env.step(env.action_space.sample())
env.render()
# custom params
env = gym.make('AX_12-v0', min_size=3, prob_r=0.5)
```

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

## Saccade
The Saccade-Antisaccade task consists in a sequence of multi-step trials where the final goal is to direct the eye movement according to previous cues projected on a screen.
The cues are essentialy of two types:
- fixation mark : squaree at the center of the screen; if it is black it is a pro-saccade trial (P), if it is white it is an anti-saccade one (A);
- location cue: circle at the side of the screen; it can be either at the left side (L) or at the right side (R) of the screen
The test-taker has 3 possible activities: F=front,L=left, R=right.
In case of a pro-saccade trial, the eye movement has to be in the same direction as the location cue (PL or PR); otherwise, it has to be in the opposite direction (AL or AR).
Each trial is composed by different phases:
- START: empty screen
- FIX: fixation mark appears; the test-taker has to fix it for two consecutive timesteps to have a first reward r_f (F selected twice)
- CUE: location cue appears together with fixation mark
- DELAY: location cue disappears for two timesteps to test the memory delay
- GO: fixation mark disappears as well (empty screen) and the subject has to solve the task (it has up to 8 timesteps to answer L or R)

e.g.  
Start: 
Screen: None

Fix:
Screen: P; Target_action: F
Screen: P; Target_action: F

Cue:
Screen: PL; Target_action: F

Dealy:
Screen: P; Target_action: F

Go:
Screen: None; Target_action: L

### Usage
```python
import gym
import gym_cog_ml_tasks

env = gym.make('Saccade-v0')
env.reset()
env.step(env.action_space.sample())
env.render()
# custom params
env = gym.make('Saccade-v0', go_reward = 5)
```

