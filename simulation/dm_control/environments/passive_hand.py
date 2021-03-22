from dm_control.rl import control
# from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import containers
from . import rotations
import os
import numpy as np
import collections
from . import base, utils
from dm_env import specs

_DEFAULT_TIME_LIMIT = 15
MODEL_XML_PATH = os.path.join('passive_hand', 'lift.xml')
_N_SUBSTEPS = 20
OBJECT_INITIAL_HEIGHT = 0.46163282

SUITE = containers.TaggedTasks()
# mjlib = mjbindings.mjlib

def _load_physics(model_path):
    if model_path.startswith('/'):
        fullpath = model_path
    else:
        fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
    if not os.path.exists(fullpath):
        raise IOError('File {} does not exist'.format(fullpath))
    return Physics.from_xml_path(fullpath)

@SUITE.add()
def lift_sparse(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    physics = _load_physics(MODEL_XML_PATH)
    task = Lift(sparse=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs, n_sub_steps=_N_SUBSTEPS)

class Physics(base.Physics):
    def grip_position(self):
        return self.named.data.site_xpos['robot0:grip']

    def grip_velocity(self):
        return self.get_site_vel('robot0:grip', False)[3:]

    def grip_rotation(self):
        return self.named.data.site_xmat['robot0:grip'].reshape((3,3))

    def object_position(self):
        return self.named.data.site_xpos['object0']

    def object_velocity(self):
        return self.get_site_vel('object0', False)[3:]

    def object_angular_velocity(self):
        return self.get_site_vel('object0', False)[:3]

class Lift(control.Task):
    def __init__(self, sparse, random=None):
        if not isinstance(random, np.random.RandomState):
            random = np.random.RandomState(random)
        self._random = random
        self._goal = np.asarray([1.46177789, 0.74909766, 0.7])
        self.gripper_extra_height = 0.2
        self.reward_type = sparse
        self.initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,

            # I FOUND THE BUG :DDD 5H OF SEARCHING OMGGGG
            # PLS NEVER HARDCODE AGAIN :'(((((((((((
            # 'object0:joint': 0.0,
        }
        self.last_reward = 0

    def _goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    @property
    def random(self):
        """Task-specific `numpy.random.RandomState` instance."""
        return self._random

    def action_spec(self, physics):
        """Returns a `BoundedArraySpec` matching the `physics` actuators."""
        return specs.BoundedArray(shape=(5,), dtype=np.float, minimum=-1., maximum=1.)

    def initialize_episode(self, physics):
        """Resets elements to their initial position
        Args:
          physics: An instance of `mujoco.Physics`.
        """
        self._physics_setup(physics,self.initial_qpos)
        physics.forward()

    def _physics_setup(self, physics, initial_qpos):
        for name, value in initial_qpos.items():
            physics.named.data.qpos[name] = value
        utils.reset_mocap_welds(physics)
        physics.forward()

        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + physics.grip_position()

        # commenting out the line below to stop the console spam
        # print('Ideal Start Position: ', gripper_target)

        gripper_rotation = np.array([1., 0., 0., 0.])
        # gripper_rotation = np.array([0,0,0,1])
        physics.named.data.mocap_pos['robot0:mocap'] = gripper_target
        physics.named.data.mocap_quat['robot0:mocap'] = gripper_rotation
        for _ in range(50):
            physics.step()
        # mjlib.mj_inverse(physics.model.ptr, physics.data.ptr)
        # print(physics.data.qfrc_inverse)

    def before_step(self, action, physics):
        """Sets the control signal for the actuators to values in `action`."""
        # Support legacy internal code.
        action = getattr(action, "continuous_actions", action)
        assert action.shape == (5,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl = action[:3]

        # arm origin at [[0.725, 0.74910034]]
        arm_origin = np.asarray([0.725, 0.74910034])
        vert_angle = action[3]
        hor_angle = -np.arctan2(*(physics.data.mocap_pos[0][:2] - arm_origin)) + np.pi / 2
        twist_angle = action[4]

        rot_ctrl = rotations.quat_mul(rotations.quat_mul(rotations.euler2quat([0., 0., hor_angle]),
                                        rotations.euler2quat([0., vert_angle, 0.])),
                                        rotations.euler2quat([twist_angle, 0., 0.]))

        pos_ctrl *= 0.05  # limit maximum change in position
        action = np.concatenate([pos_ctrl, rot_ctrl])
        utils.mocap_set_action(physics, action)

    def after_step(self, physics):
        pass

    def get_reward(self, physics):
        grip_vel = physics.get_site_vel('robot0:grip', False)
        grip_velp = grip_vel[3:]
        object_vel = physics.get_site_vel('object0', False)
        object_velp = object_vel[3:]
        object_rel_velp = object_velp - grip_velp
        object_pos = physics.object_position()
        dist = np.sum(object_rel_velp ** 2) ** (1 / 2)  # euclidian distance
        height = object_pos[2] - OBJECT_INITIAL_HEIGHT
        # score = prev + how close arm is to object + how high the object is
        self.last_reward = self.last_reward + (-dist) + height  # add previous reward, -ve so small distances win
        return self.last_reward

    def get_observation(self, physics):
        grip_pos = physics.grip_position()
        grip_vel = physics.get_site_vel('robot0:grip', False)
        grip_velp = grip_vel[3:]
        grip_velr = grip_vel[:3]
        grip_rot = rotations.mat2euler(physics.grip_rotation())

        object_pos = physics.object_position()
        object_vel = physics.get_site_vel('object0', False)
        object_velp = object_vel[3:]
        object_velr = object_vel[:3]
        object_rel_pos = object_pos - grip_pos
        object_rel_velp = object_velp - grip_velp

        obs = collections.OrderedDict()

        obs['grip_pos'] = grip_pos
        obs['grip_velp'] = grip_velp
        obs['grip_velr'] = grip_velr
        obs['grip_rot'] = grip_rot
        obs['object_pos'] = object_pos
        obs['object_rel_pos'] = object_rel_pos
        obs['object_velp'] = object_velp
        obs['object_velr'] = object_velr
        obs['object_rel_velp'] = object_rel_velp
        obs['simulation_time'] = physics.data.time

        return obs