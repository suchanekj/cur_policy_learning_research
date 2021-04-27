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

from config import *
from utility import SensorsReading


def temp_reward_func(last_reward: float, step: int, last_step: bool, readings: SensorsReading) -> float:
    dist = np.sum(readings.object_rel_pos ** 2)  # euclidean distance
    relv = np.sum(readings.object_rel_velp ** 2)  # relative velocity
    height = readings.object_pos[2]

    # -dist: smaller dist is better
    # -relv: smaller relative velocity is better
    # height: object move higher is better
    print(dist, relv)
    score = last_reward - dist - 1e3*relv
    return score


def solve(simulation_api: SimulationAPI, reward_threshold: float, timeout_s: float = 60, num_hof=1):
    """
    Run a genetic algorithm on the current simulation_control scenario until reward exceeds the threshold or timeout is reached
    """
    # need to have a reward function, and to implement timeout termination and when reward exceeds threshold
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    def evalReward(individual):
        """
        :param individual: list of elements
        """
        pb.update(1)
        pos = np.array(individual).reshape((NUM_STEPS, INPUT_SIZE))
        simulation_api.reset()
        simulation_api.specify_reward_function(temp_reward_func)
        try:  # there's a chance we generate a nonphysical sequence, so handle that exception
            score = simulation_api.run(pos)
        except:
            score = -math.inf
        return score,

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)  # COULDDO heuristic init
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=INPUT_SIZE * NUM_STEPS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalReward)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

    toolbox.register("select", tools.selTournament, tournsize=4)

    random.seed(64)

    pop = toolbox.population(n=HOF_POPULATIONS)

    hof = tools.HallOfFame(num_hof, similar=np.array_equal)

    # final is a list of all the elements in the last iteration (I think)

    final = algorithms.eaSimple(pop, toolbox, verbose=
    False, cxpb=0.5, mutpb=0.1, ngen=0 , halloffame=hof)
    # # code below is to check the output of the HOF individual
    hof_np = np.array(hof)

    # Note: hof returns an array of successful values!
    return hof_np


def evaluate(individual):
    simulation_api = SimulationAPI()
    pos = np.array(individual).reshape((NUM_STEPS, INPUT_SIZE))-0.5
    simulation_api.reset()
    simulation_api.specify_reward_function(temp_reward_func)
    score = simulation_api.run([a for i in range (10) for a in pos])
    return score


def save_to_file(hof, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    path += f'hof_it={HOF_ITERATIONS}_pop={HOF_POPULATIONS}_steps={NUM_STEPS}_time={time()}'
    with open(path, 'w') as f:
        for v in hof:
            f.write(f"{v-0.5} ")


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
