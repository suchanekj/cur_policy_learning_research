import math
import pathlib
import random
from time import time
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from tqdm import tqdm
from functools import partial

from config import *
from reward_functions import temp_reward_func


def evaluate(individual, simulation_api=None, pb=None):
    if pb is not None:
        pb.update(1)
    if simulation_api is None:
        simulation_api = SimulationAPI()
    pos = np.array(individual).reshape((NUM_STEPS, INPUT_SIZE))  # - 0.5
    simulation_api.reset()
    simulation_api.specify_reward_function(temp_reward_func)
    try:
        score = simulation_api.run([a for _ in range(10) for a in pos])
    except:
        score = -math.inf
    return score,


def solve(simulation_api: SimulationAPI, reward_threshold: float, timeout_s: float = 60, num_hof=1):
    """
    Run a genetic algorithm on the current simulation_control scenario until reward exceeds the threshold or timeout is reached
    """
    # need to have a reward function, and to implement timeout termination and when reward exceeds threshold
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    # def evalReward(individual):
    #     """
    #     :param individual: list of elements
    #     """
    #     pos = np.array(individual).reshape((NUM_STEPS, INPUT_SIZE))
    #     simulation_api.reset()
    #     simulation_api.specify_reward_function(temp_reward_func)
    #     try:  # there's a chance we generate a nonphysical sequence, so handle that exception
    #         score = simulation_api.run(pos)
    #     except:
    #         score = -math.inf
    #     return score,

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.randn)  # COULDDO heuristic init
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=INPUT_SIZE * NUM_STEPS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", partial(evaluate, simulation_api=simulation_api, pb=pb))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

    toolbox.register("select", tools.selTournament, tournsize=4)

    random.seed(64)

    pop = toolbox.population(n=HOF_POPULATIONS)

    hof = tools.HallOfFame(num_hof, similar=np.array_equal)

    # final is a list of all the elements in the last iteration (I think)

    final = algorithms.eaSimple(pop, toolbox, verbose=False, cxpb=0.5, mutpb=0.1, ngen=0, halloffame=hof)
    # # code below is to check the output of the HOF individual
    hof_np = np.array(hof)

    # Note: hof returns an array of successful values!
    return hof_np


def save_to_file(hof, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    path += f'hof_it={HOF_ITERATIONS}_pop={HOF_POPULATIONS}_steps={NUM_STEPS}_time={time()}'

    hof = np.asarray(
        [a for _ in range(10) for a in np.array(hof).reshape((NUM_STEPS, INPUT_SIZE))]
    ).reshape(NUM_STEPS * INPUT_SIZE * 10)

    with open(path, 'w') as f:
        for v in hof:
            f.write(f"{v} ")  # v - 0.5


def load_hof(path):
    print(f'loading {path}')
    with open(path, 'r') as f:
        return np.array(list(map(float, f.readlines()[0].split())))


if __name__ == '__main__':
    pb = tqdm(total=HOF_POPULATIONS + HOF_POPULATIONS * HOF_ITERATIONS // 20)  # magic number to estimate duration
    simulation_api = SimulationAPI()
    hof_list = solve(simulation_api, 0, 0, num_hof=HOF_COUNT)
    for hof in hof_list:
        save_to_file(hof, HOF_OUTPUT_DIRECTORY)
