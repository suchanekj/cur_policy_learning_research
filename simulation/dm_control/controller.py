from abc import ABC, abstractmethod
from queue import Queue
from typing import List

import numpy as np
import scipy.interpolate
import utility

class Trajectory(ABC):
    @abstractmethod
    def eval(self, t):
        pass

    @abstractmethod
    def add_state(self, pos=None, vert_rot=None, twist_rot=None, state=None):
        pass

    @abstractmethod
    def target_state(self):
        pass

    @abstractmethod
    def generate_trajectory(self, current_state):
        pass

class SplineTrajectory(Trajectory):
    def __init__(self, states: List[np.ndarray] = []):
        self.states = states
        self.tck, self.u = None, None

    def add_state(self, pos=None, vert_rot=None, twist_rot=None, state=None):
        if state is not None:
            self.states.append(state)
        elif pos is not None and vert_rot is not None and twist_rot is not None:
            self.states.append(np.concatenate([pos, vert_rot, twist_rot], axis=None))

    def eval(self, t):
        t = utility.clamp(t, 0, 1)
        return scipy.interpolate.splev(t, self.tck)

    def generate_trajectory(self, current_state):
        states = np.insert(self.states, 0, current_state, axis=0)
        deg = min(len(states) - 1 , 3)
        self.tck, self.u = scipy.interpolate.splprep(states.T, ub=0, ue=1, s=0, k=deg)

    def target_state(self):
        return self.states[-1]

class LinearTrajectory(Trajectory):
    def __init__(self, states: List[np.ndarray] = []):
        self.states = states
        self.dirs = None
        self.lengths = None
        self.times = []

    def add_state(self, pos=None, vert_rot=None, twist_rot=None, state=None):
        if state is not None:
            self.states.append(state)
        elif pos is not None and vert_rot is not None and twist_rot is not None:
            self.states.append(np.concatenate([pos, vert_rot, twist_rot], axis=None))

    def eval(self, t):
        t = utility.clamp(t, 0, 1)
        index = np.where(t <= self.times)[0]
        if len(index) > 0:
            index = index[0]
        else:
            return self.target_state()
        t_local = t / self.times[index]
        return self.states[index] + self.dirs[index] * self.lengths[index] * t_local

    def generate_trajectory(self, current_state):
        self.states.insert(0, current_state)
        states = np.array(self.states)
        self.dirs = np.diff(states, axis=0)
        self.lengths = np.linalg.norm(self.dirs, axis=1).reshape((-1,))
        self.dirs /= self.lengths[:, np.newaxis]
        total_length = np.sum(self.lengths)
        self.times = np.cumsum(self.lengths / total_length)

    def target_state(self):
        return self.states[-1]

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
            t = (time - self.start_time) / self.target_time
        if self.current_trajectory is None or (1 - t < 1e-2 and np.isclose(current_state, self.current_trajectory.target_state(), rtol=0.05, atol=1e-2).all()):
            if self.trajectories.empty():
                return self.current_trajectory.target_state()
            data = self.trajectories.get()
            self.current_trajectory = data[0]
            self.current_trajectory.generate_trajectory(current_state)
            self.target_time = data[1]
            self.reset_time(time)
        res = self.current_trajectory.eval(t)
        # print('Current State:', current_state)
        # print('Target State:', res)
        # print('Normalized Time:', t)
        # print('Condition Table:',  np.isclose(current_state, self.current_trajectory.target_state(), rtol=0.01, atol=5e-3))
        return res

    def get_action(self, readings: utility.SensorsReading):
        current_state = np.concatenate([readings.grip_pos, readings.grip_rot[1], readings.grip_rot[0]], axis=None)
        target_state = self.get_target_state(readings.simulation_time, current_state)

        if target_state is None:
            return utility.to_action(np.zeros(3), readings.grip_rot[2], readings.grip_rot[0])
        action = utility.to_action(target_state[:3] - readings.grip_pos, target_state[3], target_state[4])
        # print('Action:', action)
        return action





