import gym
from gym.envs.registration import registry, register, make, spec

# from paramatrize import Paramatrizer

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
for _ in range(1000):
    env.render()
    print(env.sim.data.sensordata) # height of object (ground is at about -0.14), 3D force by the cylinder, last is verical
    env.step(env.action_space.sample()) # take a random action
env.close()


env.sim.model.body_mass[-4] = 100 # this changes the model in this session
# ie. changes preserved even after env.reset(), but doesn't change contents in xml files
env.reset()

# prm = Paramatrizer(env)
# prm.create_xml(32)
# for i in range(3):
#     print(prm.change_object_pos())
#     env.reset()
    # for _ in range(100):
    #     env.render()
    #     o,r,d,i = env.step(env.action_space.sample()) # take a random action

env.close()
