from typing import Callable, Union
from inspect import signature
from pathlib import Path
from Parameterizer import Parameterizer

from reward_functions import placeholder_reward_func


class EnvironmentParametrization():
    def __init__(self,
                 parameters: list
    ):
        self.parameters = parameters


class SensorsReading():
    def __init(self,
               readings:dict
    ):
        self.readings = readings


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
        self.environmental_parametrization = None
        self.time_step = None
        self.reset()

    def reset(self, randomize=False):
        if(randomize):
            # TODO: split object and hand randomization into 2 functions!!!!
            pm = Parameterizer()
            self.environmental_parametrization = EnvironmentParametrization(pm.randomize_all(hand_randomization_multiplier))
            pm.export_XML()

        self.env.reset()

    def export_parameters(self) -> EnvironmentParametrization:
        return self.environmental_parametrization

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

        self.time_step = env.step(action)

    def get_current_reward(self):
        return time_step.reward

    def get_sensors_reading(self) -> SensorsReading:
        return SensorsReading(dict(time_step.observation))

    def run(self, actions) -> float:  # specify actions type
        """for open loop control"""
        raise NotImplementedError()
