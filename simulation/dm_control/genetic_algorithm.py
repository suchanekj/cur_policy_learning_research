from simulation_api import SimulationAPI

import random
import numpy as np
import time

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

def solve(simulation_api: SimulationAPI, reward_threshold: float, timeout_s: float = 60):
    """
    Run a genetic algorithm on the current simulation scenario until reward exceeds the threshold or timeout is reached
    """
     # need to have a reward function, and to implement timeout termination and when reward exceeds threshold  
        
    num_step = 5  #number of actions each individual takes
    N_SUBSTEPS = 3

    pos = np.zeros((num_step, 5))
    sens_data = [0]*num_step

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    def evalReward(individual):
        simulation_api.reset()
        score = 0
        for i in range(num_step):
            pos[i] = individual[5 * i:5 * (i + 1)] - 0.5
        for i in range(num_step):
            for j in range(N_SUBSTEPS):
                simulation_api.step(pos[i])
            sens_data[i] = simulation_api.get_sensors_reading()

            score += simulation_api.reward_func(  # need to implement a reward function
                    last_reward=0,
                    step=0,
                    last_step=0,
                    readings=sens_data[i]
                    )

        return score,

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)  # COULDDO heuristic init
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5*num_step)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalReward)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.5)

    toolbox.register("select", tools.selTournament, tournsize=4)

    random.seed(64)

    pop = toolbox.population(n=200)

    hof = tools.HallOfFame(1, similar=np.array_equal)

    final = algorithms.eaSimple(pop, toolbox, cxpb=0, mutpb=0.05, ngen=200,
                                halloffame=hof)

    hof_np = np.array(hof)
    simulation_api.reset()
    score = 0
    for i in range(num_step):
        pos[i] = individual[5 * i:5 * (i + 1)] - 0.5
    for i in range(num_step):
        for j in range(N_SUBSTEPS):
            simulation_api.step(pos[i])
        sens_data[i] = simulation_api.get_sensors_reading()

        score += simulation_api.reward_func(  # need to implement a reward function
                last_reward=0,
                step=0,
                last_step=0,
                readings=sens_data[i]
                )
    print(score)
    print(hof_np)
    print(hof_np[0, 0:5])

    return hof_np[0, 0:5]

if __name__=='__main__':
    solve(SimulationAPI(),0,0)
