from dm_control import viewer
import environments
import os
from config import *
from simulation_control.dm_control.genetic_algorithm import load_hof, evaluate

hof = load_hof(path=f'{HOF_OUTPUT_DIRECTORY}{os.listdir(HOF_OUTPUT_DIRECTORY)[-1]}')\
    .reshape(NUM_STEPS, INPUT_SIZE)
ctr = -1


def hof_policy(time_step):
    print(time_step)
    global ctr
    # moves until last action, then keeps doing the last action
    if ctr < len(hof) - 1:
        ctr += 1
    return hof[ctr]

sapi = SimulationAPI()
sapi.rebuild_XML()
env = environments.load(domain_name='passive_hand', task_name='lift_sparse')

viewer.launch(env, policy=hof_policy)
# viewer.launch(env, policy=lambda x:np.zeros((5,)))
