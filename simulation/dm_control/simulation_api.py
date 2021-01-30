from typing import Callable, Union
from inspect import signature
from pathlib import Path

from reward_functions import placeholder_reward_func


class EnvironmentParametrization():
    pass #TODO: data-structure to store information about scenario


class SensorsReading():
    pass #TODO: data-structure to store sensor readings (positions, forces...)")


class SimulationAPI:
    def __init__(
            self,
            object_randomization_multiplier: float = 1,
            hand_randomization_multiplier: float = 1,
            domain_name: str = 'passive_hand',
            task_name: str = 'lift_sparse'
    ):
        self.object_randomization_multiplier = object_randomization_multiplier
        self.hand_randomization_multiplier = hand_randomization_multiplier
        self.domain_name = domain_name
        self.task_name = task_name

        self.env = None
        self.reward_func = None
        self.specify_reward_function(placeholder_reward_func)
        self.reset()

    def reset(self, randomize=False):
        raise NotImplementedError()

    def export_parameters(self) -> EnvironmentParametrization:
        """export current parametrization"""
        raise NotImplementedError()

    def import_parameters(self, parameters: EnvironmentParametrization):
        """import a parametrization"""
        raise NotImplementedError()

    def specify_reward_function(self, reward_func: Callable):
        assert signature(reward_func) == signature(placeholder_reward_func)
        self.reward_func = reward_func

    def get_action_spec(self):
        return self.env.action_spec()

    def step(self, action):  # specify action type
        """for closed loop control"""
        raise NotImplementedError()

    def get_current_reward(self):
        """for closed loop control"""
        raise NotImplementedError()

    def get_sensors_reading(self) -> SensorsReading:
        """for closed loop control"""
        raise NotImplementedError()

    def run(self, actions) -> float:  # specify actions type
        """for open loop control"""
        raise NotImplementedError()
