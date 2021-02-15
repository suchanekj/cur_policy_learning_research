import random

import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from simulation.dm_control.utility import SensorsReading
from simulation_api import SimulationAPI


def temp_reward_func(last_reward: float, step: int, last_step: bool, readings: SensorsReading) -> float:
    return 0.5


def solve(simulation_api: SimulationAPI, reward_threshold: float, timeout_s: float = 60):
    """
    Run a genetic algorithm on the current simulation scenario until reward exceeds the threshold or timeout is reached
    """
    # need to have a reward function, and to implement timeout termination and when reward exceeds threshold

    num_step = 5  # number of actions each individual takes

    pos = np.zeros((num_step, 5))

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    def evalReward(individual):
        """
        :param individual: list of 25 elements
        """
        for i in range(num_step):
            pos[i] = individual[5 * i:5 * (i + 1)] - 0.5

        simulation_api.reset()
        simulation_api.specify_reward_function(temp_reward_func)
        score = simulation_api.run(pos)
        return score,

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)  # COULDDO heuristic init
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5 * num_step)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalReward)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.5)

    toolbox.register("select", tools.selTournament, tournsize=4)

    random.seed(64)

    pop = toolbox.population(n=200)

    # Note: hof returns an array of successful values! So its 1 item in the array
    hof = tools.HallOfFame(1, similar=np.array_equal)

    final = algorithms.eaSimple(pop, toolbox, cxpb=0, mutpb=0.05, ngen=5, halloffame=hof)

    # code below is to check the output of the HOF individual
    hof_np = np.array(hof)
    simulation_api.reset()
    score, = evalReward(hof_np.flatten())
    print(score)
    print(hof_np)
    print(hof_np[0, 0:5])

    return hof_np[0, 0:5]


if __name__ == '__main__':
    solve(SimulationAPI(), 0, 0)
