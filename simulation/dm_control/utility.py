from collections import OrderedDict
import numpy as np

# Wrapper classes
class EnvironmentParametrization():
    def __init__(self,
                 parameters: dict
                 ):
        self.object_translate = parameters['object_translate']
        self.object_change_slope = parameters['object_change_slope']
        self.robot_change_finger_length = parameters['robot_change_finger_length']
        self.robot_change_joint_stiffness = parameters['robot_change_joint_stiffness']
        self.robot_change_finger_spring_default = parameters['robot_change_finger_spring_default']
        self.robot_change_thumb_spring_default = parameters['robot_change_thumb_spring_default']
        self.robot_change_friction = parameters['robot_change_friction']

    def to_dict(self) -> dict:
        d = {}
        d['object_translate'] = self.object_translate
        d['object_change_slope'] = self.object_change_slope
        d['robot_change_finger_length'] = self.robot_change_finger_length
        d['robot_change_joint_stiffness'] = self.robot_change_joint_stiffness
        d['robot_change_finger_spring_default'] = self.robot_change_finger_spring_default
        d['robot_change_thumb_spring_default'] = self.robot_change_thumb_spring_default
        d['robot_change_friction'] = self.robot_change_friction
        return d


class SensorsReading():
    def __init__(self,
                 observation: OrderedDict
                 ):
        self.grip_pos = observation['grip_pos']
        self.grip_velp = observation['grip_velp']
        self.grip_velr = observation['grip_velr']
        self.grip_rot = observation['grip_rot']
        self.object_pos = observation['object_pos']
        self.object_rel_pos = observation['object_rel_pos']
        self.object_velp = observation['object_velp']
        self.object_velr = observation['object_velr']
        self.object_rel_velp = observation['object_rel_velp']
        self.simulation_time = observation['simulation_time']

    def to_dict(self) -> dict:
        d = {}
        d['grip_pos'] = self.grip_pos
        d['grip_velp'] = self.grip_velp
        d['grip_velr'] = self.grip_velr
        d['grip_rot'] = self.grip_rot
        d['object_pos'] = self.object_pos
        d['object_rel_pos'] = self.object_rel_pos
        d['object_velp'] = self.object_velp
        d['object_velr'] = self.object_velr
        d['object_rel_velp'] = self.object_rel_velp
        return d

        # observation=OrderedDict([('grip_pos', array([0.81951357, 0.34050714, 0.51620168])),
        # ('grip_velp', array([-1.37942599,  0.23459532, -0.04909928])),
        # ('grip_velr', array([-5.98888446,  1.09242198, -1.64533542])),
        # ('grip_rot', array([-0.03900035,  0.47753446, -1.45133045])),
        # ('object_pos', array([1.46266798, 0.75014693, 0.46163282])),
        # ('object_rel_pos', array([ 0.64315441,  0.40963979, -0.05456886])),
        # ('object_velp', array([0.00000000e+00, 0.00000000e+00, 4.66736695e-15])),
        # ('object_velr', array([0., 0., 0.])),
        # ('object_rel_velp', array([ 1.37942599, -0.23459532,  0.04909928]))]))

def clamp(x, min_val, max_val):
    return max(min_val, min(x, max_val))

def to_action(pos, vert_angle, twist_angle):
    return np.concatenate([pos, vert_angle, twist_angle], axis=None)
