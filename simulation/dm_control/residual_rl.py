from dm_control import viewer
from dm_env import TimeStep, Environment

import environments
import utility
import numpy as np
from controller import RobotController, SplineTrajectory, LinearTrajectory
from simulation_api import SimulationAPI

# env = environments.load(domain_name='passive_hand', task_name='lift_sparse', task_kwargs={'time_limit': float('20')})  # type: Environment
simulation_api = SimulationAPI()
simulation_api.reset(task_parameters={'time_limit': float('20')})

controller = RobotController()
# trajectory = SplineTrajectory()
# trajectory.add_state(pos=[1.25, 0.74909766, 0.46112417], vert_rot=0, twist_rot=0)
# trajectory.add_state(pos=[1.25, 1, 0.4], vert_rot=0, twist_rot=0)
# trajectory.add_state(pos=[1.4, 1., 0.4], vert_rot=0, twist_rot=0)
# # trajectory.add_state(pos=[1.46177789, 0.84909766, 0.46112417], vert_rot=0, twist_rot=-1.57)
# controller.add_trajectory(trajectory, 15)
liftTrajectory = LinearTrajectory()
liftTrajectory.add_state(pos=[1.25, 0.74909766, 0.46112417], vert_rot=0, twist_rot=0)
liftTrajectory.add_state(pos=[1.25, 1, 0.4], vert_rot=0, twist_rot=0)
liftTrajectory.add_state(pos=[1.4, 1., 0.4], vert_rot=0, twist_rot=0)
liftTrajectory.add_state(pos=[1.4, 0.74909766, 0.4], vert_rot=0, twist_rot=0)
# liftTrajectory.add_state(pos=[1.4, 0.74909766, 0.4], vert_rot=0, twist_rot=-0.7)
liftTrajectory.add_state(pos=[1.4, 0.74909766, 1], vert_rot=0, twist_rot=0)
controller.add_trajectory(liftTrajectory, 12)

def controller_policy(time_step: TimeStep):
    readings = utility.SensorsReading(time_step.observation)
    return controller.get_action(readings)


viewer.launch(simulation_api.env, policy=controller_policy)
