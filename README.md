# Cognitive and ML Tasks
This repo includes several tasks/environments built on [openai-gym](https://gym.openai.com/) for training RL agents. 

The original paper: [Martinolli, Marco, Wulfram Gerstner, and Aditya Gilra. ‘Multi-Timescale Memory Dynamics Extend Task Repertoire in a Reinforcement Learning Network With Attention-Gated Memory’. Frontiers in Computational Neuroscience 12 (2018).](https://doi.org/10.3389/fncom.2018.00050)

# Install
```
$ git clone https://github.com/CgnRLAgent/cog_ml_tasks.git
$ cd cog_ml_tasks
$ pip install -e .
```
**Requirements:** see the [setup.py](setup.py).

# Envs/Tasks Descriptions
1. [12_AX](gym_cog_ml_tasks/envs/ax_tasks/ax_12_env.md)
2. [AX_CPT](gym_cog_ml_tasks/envs/ax_tasks/ax_cpt_env.md)
3. [12_AX_S](gym_cog_ml_tasks/envs/ax_tasks/ax_s_12_env.md)
4. [12_AX_CPT](gym_cog_ml_tasks/envs/ax_tasks/ax_12_cpt_env.md)
5. [Simple_Copy](gym_cog_ml_tasks/envs/copy_tasks/simple_copy_env.md)
6. [Simple_Copy_Repeat](gym_cog_ml_tasks/envs/copy_tasks/copy_repeat_env.md)
7. [Sequence Prediction](gym_cog_ml_tasks/envs/task_seq_prediction/seq_prediction_env.md)
8. [Saccade](gym_cog_ml_tasks/envs/saccade_task/saccade_env.md)
