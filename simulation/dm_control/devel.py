import environments
import numpy as np
from dm_control import viewer

from simulation.dm_control.Parameterizer import Parameterizer

env = environments.load(domain_name='passive_hand', task_name='lift_sparse')
action_spec = env.action_spec()

def random_policy(time_step):
    return np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)


pm = Parameterizer()
# pm.object_translate(0.3)
pm.randomize_all(0.2)
pm.export_XML()

viewer.launch(env, policy=random_policy)