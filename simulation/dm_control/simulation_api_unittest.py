import numpy as np

from simulation.dm_control.simulation_api import SimulationAPI
from simulation.dm_control.utility import EnvironmentParametrization
from simulation.dm_control.utility import SensorsReading

# Check if simulation API works with a given step input
sapi = SimulationAPI()
sapi.step(np.array([0, 0, 0, 0, 0], dtype='float64'))
print(sapi.get_sensors_reading().grip_velp)
print(sapi.export_parameters().object_translate)

# Check if simulation API accepts a manual input
t = {
    'object_translate': 6.9,
    'object_change_slope': 0.0,
    'robot_change_finger_length': 0.0,
    'robot_change_joint_stiffness': 0.0,
    'robot_change_finger_spring_default': 0.0,
    'robot_change_thumb_spring_default': 0.0,
    'robot_change_friction': 0.0
}
ep = EnvironmentParametrization(t)
sapi.import_parameters(ep)
print(sapi.export_parameters().object_translate)

# Check if simulation API's run function works
x = np.zeros(shape=(10, 5))


def lmao(last_reward: float, step: int, last_step: bool, readings: SensorsReading) -> float:
    return 0.5


sapi.specify_reward_function(lmao)
reward = sapi.run(x)
print(reward)
