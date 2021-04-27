import numpy as np

from simulation.dm_control.utility import SensorsReading


def placeholder_reward_func(last_reward: float, step: int, last_step: bool, readings: SensorsReading) -> float:
    raise NotImplementedError()


def temp_reward_func(last_reward: float, step: int, last_step: bool, readings: SensorsReading) -> float:
    dist = np.sum(readings.object_rel_pos ** 2)  # euclidean distance
    relv = np.sum(readings.object_rel_velp ** 2)  # relative velocity
    height = readings.object_pos[2]

    # -dist: smaller dist is better
    # -relv: smaller relative velocity is better
    # height: object move higher is better
    print(dist, relv)
    score = last_reward - dist - 1e3*relv
    return score
