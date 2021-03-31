# TODO: port reward function from ../mujoco_gym/hand_optimization.py
# from simulation_control.dm_control.simulation_control import SensorsReading
from utility import SensorsReading


def placeholder_reward_func(last_reward: float, step: int, last_step: bool, readings: SensorsReading) -> float:
    raise NotImplementedError()