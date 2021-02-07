# TODO: port reward function from ../mujoco_gym/hand_optimization.py
# from simulation.dm_control.simulation_api import SensorsReading
from simulation.dm_control.utility import SensorsReading


def placeholder_reward_func(last_reward: float, step: int, last_step: bool, readings: SensorsReading) -> float:
    raise NotImplementedError()