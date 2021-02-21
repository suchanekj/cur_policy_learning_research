import environments
import numpy as np
from dm_control import viewer
from parameterizer import Parameterizer
from simulation.dm_control.genetic_algorithm import solve
from simulation.dm_control.simulation_api import SimulationAPI

env = environments.load(domain_name='passive_hand', task_name='lift_sparse')
# action_spec = env.action_spec()
hof = solve(SimulationAPI(), 0, 0).reshape((5,-1))
ctr: int = -1

def hof_policy(time_step):
    global ctr
    ctr = (ctr + 1) % 5
    # print(time_step)
    # print(time_step.observation)
    print(hof[ctr])
    return hof[ctr]


# pm = Parameterizer()
# pm.object_translate(0.3)
# # print(pm.randomize_all(0.2))
# pm.export_XML()

viewer.launch(env, policy=hof_policy)
# time_step = env.step(action_spec)
# print(time_step.observation['grip_pos'])
# print(time_step)
# print(time_step.reward)

