import gym
import passive_hand_env
from gym.envs.registration import registry, register, make, spec



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