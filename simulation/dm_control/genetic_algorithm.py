# from simulation_api import SimulationAPI
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import numpy as np
from numpy import random

# TODO: port ../mujoco_gym/hand_optimization.py
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
IND_SIZE = 25
toolbox.register("attr_float", random.random)  # COULDDO heuristic init
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalPos)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.5)

toolbox.register("select", tools.selTournament, tournsize=4)

def ea():
    random.seed(64)

    pop = toolbox.population(n=200)

    hof = tools.HallOfFame(1, similar=np.array_equal)

    final = algorithms.eaSimple(pop, toolbox, cxpb=0, mutpb=0.05, ngen=200,
                                halloffame=hof)

    return hof


def evalPos(individual):
    pass


# def solve(simulation_api: SimulationAPI, reward_threshold: float, timeout_s: float = 60):
#     """
#     Run a genetic algorithm on the current simulation scenario until reward exceeds the threshold or timeout is reached
#     """
#     raise NotImplementedError()
