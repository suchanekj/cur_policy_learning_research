from inspect import signature
from typing import Callable

import environments
from parameterizer import Parameterizer
from reward_functions import placeholder_reward_func
from utility import *


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

        self.reward = 0
        self.step_index = 0
        self.env = None
        self.reward_func = None
        self.specify_reward_function(placeholder_reward_func)
        self.environmental_parametrization = None
        self.time_step = None
        self.reset()

    def reset(self, randomize=False, parameters: EnvironmentParametrization = None, task_parameters: dict = None):
        """
        Resets simulation. After a run the hand will be in some weird position so
        you need to call this function to put it back into place. And when resetting
        you can also choose to randomise the parameters of the simulation as well.

        :param randomize: whether to randomize the inputs
        :param parameters: set simulation to a specific set of parameters
        :param task_parameters: parameters to pass to task initialization
        """
        pm = Parameterizer()
        if randomize or (parameters is not None):
            if (randomize):
                pm.randomize_object(self.object_randomization_multiplier)
                pm.randomize_robot(self.hand_randomization_multiplier)
            if parameters is not None:
                pm.set_all(parameters.to_dict())
            pm.export_XML()
        para_dict = pm.get_parameters()
        self.environmental_parametrization = EnvironmentParametrization(para_dict)
        self.env = environments.load(domain_name=self.domain_name, task_name=self.task_name,
                                     task_kwargs=task_parameters)
        self.env.reset()
        self.reward = 0
        self.step_index = 0
    def rebuild_XML(self):
        """
        rebuilds XML files in passive_hand from XML files in passive_hand_unmodified
        """
        pm = Parameterizer()
        pm.export_XML()

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

    def step(self, action):  # TODO: specify action type
        """
        for closed loop control
        """
        self.time_step = self.env.step(action)

    def get_current_reward(self):
        return self.time_step.reward

    def get_sensors_reading(self) -> SensorsReading:
        assert self.time_step is not None
        return SensorsReading(self.time_step.observation)

    def run(self, actions) -> float:  # TODO: specify actions type
        """
        Runs the simulation using a sequence of actions that the robot should take.

        :param actions:
            list of actions
            each action should have the same shape as the action space
        """
        for i, action in enumerate(actions):
            self.step(action)
            self.reward = self.reward_func(self.reward, self.step, i == len(actions) - 1, self.get_sensors_reading())
            self.step_index += 1
        return self.reward
