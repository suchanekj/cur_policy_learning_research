from queue import Queue
from typing import List

import numpy as np
import scipy.interpolate
import utility

class Trajectory:
    def __init__(self, states: List[np.ndarray] = []):
        # x, y, z = states[:, 0], states[:, 1], states[:, 2]
        self.states = states
        # self.rotation = states[:, 3:]
        self.tck, self.u = None, None

    def add_state(self, pos=None, vert_rot=None, twist_rot=None, state=None):
        if state is not None:
            self.states.append(state)
        elif pos is not None and vert_rot is not None and twist_rot is not None:
            self.states.append(np.concatenate([pos, vert_rot, twist_rot], axis=None))

    def eval(self, t):
        t = utility.clamp(t, 0, 1)
        return scipy.interpolate.splev(t, self.tck)

    def generate_spline(self, current_state):
        states = np.insert(self.states, 0, current_state, axis=0)
        deg = min(len(states) - 1 , 3)
        # print(deg, states)
        self.tck, self.u = scipy.interpolate.splprep(states.T, ub=0, ue=1, s=0, k=deg)


class RobotController:
    def __init__(self):
        self.trajectories = Queue() # type: Queue[Tuple[Trajectory, float]]
        self.start_time = 0
        self.current_trajectory = None
        self.target_time = None

    def reset_time(self, time):
        self.start_time = time

    def add_trajectory(self, trajectory: Trajectory, target_time: float):
        self.trajectories.put((trajectory, target_time))

    def get_target_state(self, time, current_state):
        t = 0
        if self.target_time is not None:
            t = utility.clamp((time - self.start_time) / self.target_time, 0, 1)
        if self.current_trajectory is None or abs(t - 1) < 1e-4:
            if self.trajectories.empty():
                return None
            data = self.trajectories.get()
            self.current_trajectory = data[0]
            self.current_trajectory.generate_spline(current_state)
            self.target_time = data[1]
            self.reset_time(time)
        return self.current_trajectory.eval(t)

    def get_action(self, readings: utility.SensorsReading):
        current_state = np.concatenate([readings.grip_pos, readings.grip_rot[2], readings.grip_rot[0]], axis=None)
        target_state = self.get_target_state(readings.simulation_time, current_state)

        if target_state is None:
            return utility.to_action(np.zeros(3), readings.grip_rot[2], readings.grip_rot[0])
        return utility.to_action(target_state[:3] - readings.grip_pos, target_state[3], target_state[4])





