# Robotics environments

These robotics environments were provided by OpenAI in the following paper:

```
@misc{1802.09464,
  Author = {Matthias Plappert and Marcin Andrychowicz and Alex Ray and Bob McGrew and Bowen Baker and Glenn Powell and Jonas Schneider and Josh Tobin and Maciek Chociej and Peter Welinder and Vikash Kumar and Wojciech Zaremba},
  Title = {Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research},
  Year = {2018},
  Eprint = {arXiv:1802.09464},
}
```

In addition, this code was largely based off of the DeepMind Control Suite developed by DeepMind as detailed in the following paper:

```
@misc{tassa2020dmcontrol,
    title={dm_control: Software and Tasks for Continuous Control},
    author={Yuval Tassa and Saran Tunyasuvunakool and Alistair Muldal and
            Yotam Doron and Siqi Liu and Steven Bohez and Josh Merel and
            Tom Erez and Timothy Lillicrap and Nicolas Heess},
    year={2020},
    eprint={2006.12983},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

##Differences
This model has been ported from use in OpenAI Gym to use in dm_control. There are some major differences in design bewteen the two.

1. In dm_control, actions are done through actuators whereas this model has none. One task would be to add actuators to the model in XML and change the code accordingly.
2. In OpenAI Gym, tasks and environments are bundled, where action spaces, rewards, and observation are all handled by the environment. In dm_control, tasks and environments are segmented such that tasks provide observations, rewards, and action spaces and environments merely execute the actions. The main data pipeline between are the Physics objects found in dm_control.mujoco.engine.
3. Pretty much everything is an OrderedDict in dm_control unlike OpenAI where everything is a flattened numpy array. Features like physics.named.data are useful in maintaining code readability. 

##Notes
To load the passive hand model, please use the loader in the passive_hand_env module. Some sample code can be found below:

```python
import environments
import numpy as np

# Load task and domain
env = environments.load(domain_name='passive_hand', task_name='lift_sparse')

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()

while not time_step.last():
    action = np.random.uniform(action_spec.minimum,
                               action_spec.maximum,
                               size=action_spec.shape)
    time_step = env.step(action)
    print(time_step.reward, time_step.discount, time_step.observation)
```
If you would like an interactive viewer, use the following sample code:
```python
import environments
import numpy as np
from dm_control import viewer

# load task and domain
env = environments.load(domain_name='passive_hand', task_name='lift_sparse')

action_spec = env.action_spec()

def random_policy(time_step):
    # Find random policy based on action_spec
    return np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)

# Launch interactive viewer
viewer.launch(env, policy=random_policy)
```