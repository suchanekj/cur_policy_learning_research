import gym
import passive_hand_env
from passive_hand_env.passive_hand_env import goal_distance
from gym.envs.registration import registry, register, make, spec
import random
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import time

kwargs = {
    'reward_type': 'sparse',
}

# Fetch
register(
    id='PassiveHandLift-v0',
    entry_point='passive_hand_env:PassiveHandLift',
    kwargs=kwargs,
    max_episode_steps=50,
)

env = gym.make('PassiveHandLift-v0')
env.reset()
print(env.action_space.sample())

cylinder_pos = np.array([1.46177789, 0.74909766, 0])
pos = np.zeros((5, 5))
fin_pos = np.zeros((5, 5))
obs = np.zeros((5, 25))
sens_data = np.zeros((5, 4))

N_SUBSTEPS = 3


def main():
    hof = ea()
    hof_np = np.array(hof)

    env.reset()
    for i in range(5):
        fin_pos[i] = hof_np[0, 5 * i:5 * (i + 1)] - 0.5  # each coodinate for postion
    for i in range(5):
        for j in range(N_SUBSTEPS):
            env.render()
            time.sleep(3 / N_SUBSTEPS)
            observation, reward, done, infto = env.step(fin_pos[i])
            sens_data[i] = env.sim.data.sensordata
            obs[i] = observation['observation']

    hof_np = np.array(hof)
    print(hof_np.shape)
    print(hof_np)
    print(hof_np[0, 0:5])

    current_pos = np.array(obs[-1][:3])

    distance_score = -goal_distance(current_pos, cylinder_pos)
    h_force_score = -sum(sum(abs(sens_data[:,1:3]))) * 0.05
    v_force_score = -sum(abs(sens_data[:,3]))
    height_score = sum(sens_data[:,0]) * 1000
    score = height_score + h_force_score + v_force_score + distance_score
    print(distance_score, h_force_score, v_force_score, height_score, score)


def evalPos(individual):
    env.reset()
    for i in range(5):
        pos[i] = individual[5 * i:5 * (i + 1)] - 0.5
        # each coodinate for postion
    for i in range(5):
        for j in range(N_SUBSTEPS):
            # env.render()
            observation, reward, done, infto = env.step(pos[i])
            sens_data[i] = env.sim.data.sensordata
            obs[i] = observation['observation']

    current_pos = np.array(obs[-1][:3])

    distance_score = -goal_distance(current_pos, cylinder_pos) * 100
    h_force_score = -sum(sum(abs(sens_data[:,1:3]))) * 0.05
    v_force_score = -sum(abs(sens_data[:,3]))
    height_score = sum(sens_data[:,0]) * 1000
    score = height_score + h_force_score + v_force_score + distance_score
    print(distance_score, h_force_score, v_force_score, height_score, score)
    # print(round(score, 2), end="\t")
    return score,


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


if __name__ == "__main__":
    main()
