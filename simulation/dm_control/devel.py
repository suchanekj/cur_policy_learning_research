import environments
import numpy as np
from dm_control import viewer
from parameterizer import Parameterizer

env = environments.load(domain_name='passive_hand', task_name='lift_sparse')
action_spec = env.action_spec()

def random_policy(time_step):
    # print(time_step)
    # print(time_step.observation)
    return np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)


pm = Parameterizer()
pm.object_translate(0.3)
# print(pm.randomize_all(0.2))
pm.export_XML()

# viewer.launch(env, policy=random_policy)
# time_step = env.step(action_spec)
# print(time_step.observation['grip_pos'])
# print(time_step)
# print(time_step.reward)
