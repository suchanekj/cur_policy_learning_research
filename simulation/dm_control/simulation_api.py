from typing import Callable, Union
from inspect import signature
from pathlib import Path
from parameterizer import Parameterizer
from utility import *
import environments
import numpy as np

from reward_functions import placeholder_reward_func


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

    def reset(self, randomize=False, parameters: EnvironmentParametrization = None):
        self.env = environments.load(domain_name=self.domain_name, task_name=self.task_name)
        pm = Parameterizer()
        if (randomize):
            pm.randomize_object(self.object_randomization_multiplier)
            pm.randomize_robot(self.hand_randomization_multiplier)
        if parameters is not None:
            pm.set_all(parameters.to_dict())
        para_dict = pm.export_XML()
        self.environmental_parametrization = EnvironmentParametrization(para_dict)
        self.env.reset()

    def export_parameters(self) -> EnvironmentParametrization:
        return self.environmental_parametrization

    def import_parameters(self, parameters: EnvironmentParametrization):
        # after a change of parameters, we have to reset the environment too!
        self.reset(parameters=parameters)

    def specify_reward_function(self, reward_func: Callable):
        assert signature(reward_func) == signature(placeholder_reward_func)
        self.reward_func = reward_func

    def get_action_spec(self):
        return self.env.action_spec()

    def step(self, action):  # specify action type
        """for closed loop control"""
        self.time_step = self.env.step(action)

    def get_current_reward(self):
        return self.time_step.reward

    def get_sensors_reading(self) -> SensorsReading:
        return SensorsReading(self.time_step.observation)

    def run(self, actions) -> float:  # specify actions type
        for action in actions:
            self.step(action)


