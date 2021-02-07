import gym
from gym.envs.registration import registry, register, make, spec
import time
import os
import numpy as np
import mujoco_py
import xml.etree.ElementTree as ET


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
env.sim.model.body_mass[-4] = 100 # this changes the model in this session 
# ie. changes preserved even after env.reset(), but doesn't change contents in xml files
env.reset()

