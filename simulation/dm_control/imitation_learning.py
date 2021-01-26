from typing import List, Union
from pathlib import Path

from simulation_api import EnvironmentParametrization, SimulationAPI


class ImitationExample:
    pass  #TODO: data-structure to store a EnvironmentParametrization and a solution and able to load and save to a file


class ImitationDataset:
    data: List[ImitationExample] = []
    object_randomization_multiplier: float = 1
    hand_randomization_multiplier: float = 0
    domain_name: str = 'passive_hand'
    task_name: str = 'lift_sparse'

    def __init__(self):
        raise NotImplementedError()

    def save(self, folder: Union[Path, str]):
        # all parameters of this class, not just "data"
        raise NotImplementedError()

    def load(self, folder: Union[Path, str]):
        raise NotImplementedError()

    def generate(self, size: int):
        # increase data to size, use GA to find solutions
        raise NotImplementedError()

    def get_random_scenario(self):
        # generate a new scenario for testing, using the same rules as used during genearion
        raise NotImplementedError()


class ImitationLearner:
    dataset: ImitationDataset
    # annotate model

    def __init__(self):  # add model parameters
        raise NotImplementedError()

    def fit(self, visualize=False):  # add training parameters
        raise NotImplementedError()

    def test(self, visualize=False):  # size of test etc.
        raise NotImplementedError()
