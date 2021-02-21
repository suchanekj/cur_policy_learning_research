import random

import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from config import *
from simulation.dm_control.utility import SensorsReading


def temp_reward_func(last_reward: float, step: int, last_step: bool, readings: SensorsReading) -> float:
    dist = np.sum(readings.object_rel_velp ** 2) ** (1 / 2)  # euclidian distance
    height = readings.object_pos[2] - OBJECT_INITIAL_HEIGHT
    # score = prev + how close arm is to object + how high the object is
    score = last_reward + (-dist) + height  # add previous reward, -ve so small distances win
    return score


def solve(simulation_api: SimulationAPI, reward_threshold: float, timeout_s: float = 60):
    """
    Run a genetic algorithm on the current simulation scenario until reward exceeds the threshold or timeout is reached
    """
    # need to have a reward function, and to implement timeout termination and when reward exceeds threshold

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    def evalReward(individual):
        """
        :param individual: list of elements
        """
        pos = np.array(individual).reshape((NUM_STEPS, INPUT_SIZE))
        simulation_api.reset()
        simulation_api.specify_reward_function(temp_reward_func)
        score = simulation_api.run(pos)
        return score,

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)  # COULDDO heuristic init
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=INPUT_SIZE * NUM_STEPS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalReward)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.5)

    toolbox.register("select", tools.selTournament, tournsize=4)

    random.seed(64)

    pop = toolbox.population(n=200)

    hof = tools.HallOfFame(1, similar=np.array_equal)

    # final is a list of all the elements in the last iteration (I think)
    final = algorithms.eaSimple(pop, toolbox, cxpb=0, mutpb=0.05, ngen=HOF_ITERATIONS, halloffame=hof)

    # # code below is to check the output of the HOF individual
    hof_np = np.array(hof)
    # simulation_api.reset()
    score, = evalReward(hof_np.flatten())

    # Note: hof returns an array of successful values! So its 1 item in the array
    return hof_np[0]


def evaluate(individual):
    simulation_api = SimulationAPI()
    pos = np.array(individual).reshape((NUM_STEPS, INPUT_SIZE))
    simulation_api.reset()
    simulation_api.specify_reward_function(temp_reward_func)
    score = simulation_api.run(pos)
    return score


def save_to_file(hof, path):
    with open(path, 'w') as f:
        for v in hof:
            f.write(f"{v} ")


def load_hof(path=HOF_OUTPUT_PATH):
    with open(path, 'r') as f:
        return np.array(list(map(float, f.readlines()[0].split())))


if __name__ == '__main__':
    simulation_api = SimulationAPI()
    hof = solve(simulation_api, 0, 0)
    save_to_file(hof, HOF_OUTPUT_PATH)
