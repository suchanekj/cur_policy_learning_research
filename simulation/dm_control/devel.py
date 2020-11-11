import passive_hand_env
import numpy as np
from dm_control import viewer

env = passive_hand_env.load(domain_name='passive_hand', task_name='lift_sparse')
action_spec = env.action_spec()

def random_policy(time_step):
    return np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)

viewer.launch(env, policy=random_policy)
