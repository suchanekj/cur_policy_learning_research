from dm_control import viewer

import environments
from config import *
from simulation.dm_control import genetic_algorithm
from simulation.dm_control.genetic_algorithm import load_hof, evaluate

env = environments.load(domain_name='passive_hand',     task_name='lift_sparse')


hof = load_hof().reshape(NUM_STEPS, INPUT_SIZE)
print(hof)
print(evaluate(hof))

ctr = -1


def hof_policy(time_step):
    print(time_step)
    global ctr
    if ctr < len(hof) - 1:
        ctr += 1
    # # print(time_step)
    # # print(time_step.observation)
    # print(hof[ctr])
    return hof[ctr]


viewer.launch(env, policy=hof_policy)
