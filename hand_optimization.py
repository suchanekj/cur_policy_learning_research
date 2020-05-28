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
pos = np.zeros((5,5))
fin_pos = np.zeros((5,5))
obs = np.zeros((5,25))
sens_data = np.zeros((5,4))



def main():
    hof = ea()
    hof_np = np.array(hof)

    env.reset()
    for i in range(5):
        fin_pos[i] = hof_np[0,5*i:5*(i+1)]  #each coodinate for postion
    for i in range(5):
        env.render()
        time.sleep(3)
        observation , reward, done, infto = env.step(fin_pos[i])
    
    hof_np = np.array(hof)
    print(hof_np.shape)
    print(hof_np)
    print(hof_np[0,0:5])
    


def evalPos(individual):
    for i in range(5):
        pos[i] = individual[5*i:5*(i+1)] 
       #each coodinate for postion
    for i in range(5):
        observation , reward, done, infto = env.step(pos[i])
        sens_data[i] = env.sim.data.sensordata 
        obs[i] = observation['observation']
        
    current_pos = np.array(obs[4][:3])
    
    # sens_data[:,0] = [0,0,0,0,0]
    recip = 5/goal_distance(current_pos,cylinder_pos)
    return sens_data.sum() + recip,
    



creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

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
    
    pop = toolbox.population(n=10)
    
    hof = tools.HallOfFame(1, similar=numpy.array_equal)
    
 
    final = algorithms.eaSimple(pop, toolbox, cxpb=0, mutpb=0.05, ngen=500, 
                        halloffame=hof)

    


    
    return hof


if __name__ == "__main__":
    main()
