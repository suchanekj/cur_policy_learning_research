from dm_control import viewer
from dm_env import TimeStep, Environment

import environments
import utility
import numpy as np
from controller import RobotController, Trajectory

env = environments.load(domain_name='passive_hand', task_name='lift_sparse')  # type: Environment
action_spec = env.action_spec()

controller = RobotController()
trajectory = Trajectory()
trajectory.add_state(pos=[1.46177789, 0.74909766, 0.46112417], vert_rot=0, twist_rot=-1.57)
controller.add_trajectory(trajectory, 4)

def random_policy(time_step: TimeStep):
    readings = utility.SensorsReading(time_step.observation)
    return controller.get_action(readings)


viewer.launch(env, policy=random_policy)
