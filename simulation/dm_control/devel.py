import environments
import numpy as np
from dm_control import viewer

env = environments.load(domain_name='passive_hand', task_name='lift_sparse')
action_spec = env.action_spec()

def random_policy(time_step):
    env.physics.named.data.xpos['object0'][2] = 1.6
    # print(env.physics.named.data.xpos)
    return np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)

print(dir(env.physics.named.data))

viewer.launch(env, policy=random_policy)

