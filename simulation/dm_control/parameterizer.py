import random
import shutil, os
import copy
import inspect

import numpy as np
import xml.etree.ElementTree as ET
import utility

class Parameterizer:
    """
    Randomizes parameters in XML.

    How to use:

    pm = Parameterizer()
    pm.randomize_all(0.2) // returns an array of random doubles used to parametrize
    pm.export_XML()

    """
    TOTAL_PARAMETERS = 7
    PARAMETER_DICT = utility.EnvironmentParametrization.DEFAULT.copy()

    xml_folder = os.path.join(os.path.dirname(__file__), 'simulation_control', 'environments', 'assets')
    unmodified_lift = os.path.join(xml_folder, 'passive_hand_unmodified', 'lift.xml')
    unmodified_robot = os.path.join(xml_folder, 'passive_hand_unmodified', 'robot.xml')
    modified_lift = os.path.join(xml_folder, 'passive_hand', 'lift.xml')
    modified_robot = os.path.join(xml_folder, 'passive_hand', 'robot.xml')

    def update_XML(self):
        FUNC_ARR = [
            self.object_translate,
            self.object_change_slope,
            self.robot_change_finger_length,
            self.robot_change_joint_stiffness,
            self.robot_change_finger_spring_default,
            self.robot_change_thumb_spring_default,
            self.robot_change_friction
        ]
        for func in FUNC_ARR:
            print('updated', func.__name__)
            func(self.PARAMETER_DICT[func.__name__])

    def set_all(self, d: dict):
        assert len(d) == len(self.PARAMETER_DICT)
        for k in d:
            assert k in self.PARAMETER_DICT
        self.PARAMETER_DICT = d

    def randomize_all(self, v):
        assert v >= 0 and v <= 1
        self.randomize_object(v)
        self.randomize_robot(v)

    def randomize_object(self, v):
        assert v >= 0 and v <= 1
        self.PARAMETER_DICT['object_translate'] = random.random() * v
        self.PARAMETER_DICT['object_change_slope'] = random.random() * v

    def randomize_robot(self, v):
        assert v >= 0 and v <= 1
        self.PARAMETER_DICT['robot_change_finger_length'] = random.random() * v
        self.PARAMETER_DICT['robot_change_joint_stiffness'] = random.random() * v
        self.PARAMETER_DICT['robot_change_finger_spring_default'] = random.random() * v
        self.PARAMETER_DICT['robot_change_thumb_spring_default'] = random.random() * v
        self.PARAMETER_DICT['robot_change_friction'] = random.random() * v

    def __init__(self):
        """Initialises an xml tree for lift.xml and robot.xml"""

        self.lift_tree = ET.parse(Parameterizer.unmodified_lift)
        self.robot_tree = ET.parse(Parameterizer.unmodified_robot)

        self.lift_root = self.lift_tree.getroot()
        self.robot_root = self.robot_tree.getroot()

    def _translate(self, pos: str, dx, dy, dz) -> str:
        """takes a position string, translates it by dxyz, and returns it"""

        xyz = np.array([float(x) for x in pos.split(' ')])
        dxyz = np.array([dx, dy, dz])
        xyz = list(xyz + dxyz)
        xyz = [str(x) for x in xyz]
        xyz = ' '.join(xyz)
        return xyz

    def object_translate(self, v):
        """
        Shifts the object randomly about its origin by a random distance within
        v * object radius
        """
        dx = 0.025 * v * self.random11()
        dy = 0.025 * v * self.random11()
        dz = 0

        object = [i for i in self.lift_root.iter('body') if i.get('name') == 'object0'][0]

        pos = object.attrib['pos']
        object.attrib['pos'] = self._translate(pos, dx, dy, dz)

        tables = [i for i in self.lift_root.iter('body') if 'table' in i.get('name')]
        for table in tables:
            pos = table.attrib['pos']
            table.attrib['pos'] = self._translate(pos, dx, dy, dz)

    # def object_change_slope(self, r=0.025, rbtm=0.03, h=0.120, t=0.012):
    def object_change_slope(self, v):

        """
        v -- Range: [0..1]

        Creates an object whose radius varies from bottom to top.
        For now, the default value is halfway between a cylinder and an upside-down cone,
        and randomising creates an object that could be closer to a cylinder or cone

        r -- middle radius
        rbtm -- bottom radius
        h -- height of objecct
        t -- thickness of each slice
        """

        r = 0.035 #0.025
        rbtm = 0.013 + self.random11() * v * 0.012
        h = 0.120
        t = 0.012
        n = int(h / t)
        dr = 2 * (rbtm - r) / n

        object = [i for i in self.lift_root.iter('body') if i.get('name') == 'object0'][0]

        for i in range(n):
            z2 = i * t - h / 2
            r2 = rbtm - i * dr
            cylinder = ET.Element('geom')
            nxtpos = ' '.join(('0', '0', str(z2)))
            nxtsz = ' '.join((str(r2), str(t / 2)))
            cylinder.attrib = dict(name='object0' + str(i), pos=nxtpos, size=nxtsz, type='cylinder', condim='3',
                                   material='block_mat', mass='0')
            object.append(cylinder)

    def robot_change_finger_length(self, v):
        """
        Randomly changes the finger length by up to v * 0.02

        I had to make the default extension 0.01, so that we can vary the finger length about
        0.01, with range = [0, 0.02]

        0.02 is hardcoded, it makes the fingers looong
        """
        extra_length = 0.01 * (1 + v * self.random11())
        finger_segment_list = []
        for element in self.robot_root.iter('body'):
            for finger_part in ["proximal", "middle", "distal"]:
                if finger_part in element.get('name'):
                    finger_segment_list.append(element)

        for finger_segment in finger_segment_list:
            finger_segment.attrib['pos'] = self._translate(finger_segment.attrib['pos'], 0, 0, extra_length)

    def robot_change_joint_stiffness(self, v):
        """
        Changes the stiffness value randomly by up to v of the original value
        """
        stiffness = 80.0 * (1 + v * self.random11())
        palm = [i for i in self.robot_root.iter('body') if i.get('name') == 'robot0:palm'][0]
        joints = [i for i in palm.iter('joint')]
        for i in joints:
            i.attrib['stiffness'] = str(stiffness)

    def robot_change_finger_spring_default(self, v):
        """
        Changes default spring value randomly by up to v of the original value

        Changes default finger position by changing the default lengths for the springs in the proximal, middle and distal finger joints.
        a is in the range [0, 1.571], where 0 is for fully extended and 1.571 is for fully bent.
        a -- default position for all three fingers (original: 0.8)
        """
        a = 0.8 * (1 + v * self.random11())
        palm = [i for i in self.robot_root.iter('body') if i.get('name') == 'robot0:palm'][0]
        for i in palm.iter('joint'):
            if not any(_ in i.attrib['name'] for _ in ["J4", "J3", "TH"]):
                i.attrib['springref'] = str(a)

    def robot_change_thumb_spring_default(self, v):
        """
        Changes default spring value randomly by up to v of the original value

        Changes default thumb position by changing the default lengths for the thumb joints.

        a -- thumb base rotation outside of the palm plane(gripping), range [-1.047, 1.047], where 1.047 is for fully gripped
        b -- thumb base rotation in the palm plane, range [0, 1.6], where 0 is when thumb is closest to the index finger(adduction)
        c -- distal knuckle bending, range [-1.571, 0], where -1.571 is for fully bent
        """
        a = 0.1 * (1 + v * self.random01())
        b = 1.6 * (1 + v * self.random01())
        c = -0.8 * (1 + v * self.random01())
        thbase = [i for i in self.robot_root.iter('body') if i.get('name') == 'robot0:thbase'][0]
        for i in thbase.iter('joint'):
            if (i.attrib['name'] == "robot0:THJ4"):
                i.attrib['springref'] = str(a)
            elif (i.attrib['name'] == "robot0:THJ3"):
                i.attrib['springref'] = str(b)
            elif (i.attrib['name'] == "robot0:THJ0"):
                i.attrib['springref'] = str(c)

    def robot_change_friction(self, v):
        """
        Changes default friction value randomly by up to v of the original value


        Changes the friction in robot's fingertips
        a -- translational friction
        b -- rotational friction
        c -- rolling friction
        """
        a = 1 * (1 + v * self.random11())
        b = 0.005 * (1 + v * self.random11())
        c = 0.0001 * (1 + v * self.random11())

        for site in ['th', 'ff', 'mf', 'rf', 'lf']:
            body = [i for i in self.robot_root.iter('body') if i.get('name') == 'robot0:' + site + 'distal'][0]
            tip = [i for i in body.iter('geom') if i.get('name') == 'robot0:C_' + site + 'distal'][0]
            f = ' '.join(map(str, [a, b, c]))
            tip.set('friction', f)

    def debug(self):
        self.printer(self.lift_root[1])

    def printer(self, e):
        print("Curr: " + e.tag, e.attrib)
        print("Children: ")
        for child in e:
            print(child.tag, child.attrib)
        print()

    def random11(self):
        return 2 * random.random() - 1

    def random01(self):
        return random.random()
    def get_parameters(self) -> dict:
        return self.PARAMETER_DICT
    def export_XML(self):
        self.update_XML()
        self.lift_tree.write(Parameterizer.modified_lift)
        self.robot_tree.write(Parameterizer.modified_robot)
        # return self.PARAMETER_DICT

# Example code:
# pm = Parameterizer()
# pm.object_translate(0.3)
# pm.randomize_all(0.2)
# pm.export_XML()