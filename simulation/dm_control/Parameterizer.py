import shutil, os
import copy
import inspect

import numpy as np
import xml.etree.ElementTree as ET

class Parameterizer:
    xml_folder = os.path.join(os.path.dirname(__file__), 'environments', 'assets')
    unmodified_lift = os.path.join(xml_folder, 'passive_hand_unmodified', 'lift.xml')
    unmodified_robot = os.path.join(xml_folder, 'passive_hand_unmodified', 'robot.xml')
    modified_lift = os.path.join(xml_folder, 'passive_hand', 'lift.xml')
    modified_robot = os.path.join(xml_folder, 'passive_hand', 'robot.xml')

    def randomize(self, i):
        """
        Randomizes the model parameters. Returns a tuple of all the randomized parameters.
        Upper bound: ~20% difference from original values (modify if necessary!)

        i -- input range: [0, 1], i.e. 0 to 1 --> 0% change to 20% change
        """
        if(i < 0 or i > 1): return None
        sig = inspect.signature(self.object_translate)
        print(sig.parameters.items())
        # object_translate(self, dx, dy, dz)
        # object_change_slope(self, r=0.025, rbtm=0.03, h=0.120, t=0.012)
        # robot_change_joint_stiffness(self, v)
        # robot_change_finger_spring_default(self, a)
        # robot_change_thumb_spring_default(self, a, b, c)
        # robot_change_friction(self, a, b, c)


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

    def object_translate(self, dx=0, dy=0, dz=0):
        """modifies lift_root by shifting the object position by xyz"""

        # needs to be updated once dhilan adds in his objects

        object = self.lift_root[4][4]
        pos = object.attrib['pos']
        object.attrib['pos'] = self._translate(pos, dx, dy, dz)

    def object_change_slope(self, r=0.025, rbtm=0.03, h=0.120, t=0.012):
        """
        Creates an object whose radius varies from bottom to top.

        r -- middle radius
        rbtm -- bottom radius
        h -- height of objecct
        t -- thickness of each slice
        """
        n = int(h / t)
        dr = 2 * (rbtm - r) / n

        object = self.lift_root[4][4]

        for i in range(n):
            z2 = i * t - h/2
            r2 = rbtm - i * dr
            cylinder = ET.Element('geom')
            nxtpos = ' '.join(('0', '0', str(z2)))
            nxtsz = ' '.join((str(r2), str(t / 2)))
            cylinder.attrib = dict(name='object0' + str(i), pos=nxtpos, size=nxtsz, type='cylinder', condim='3',
                               material='block_mat', mass='0')
            object.append(cylinder)

    def robot_change_finger_length(self, v=0):
        """
        Increases current finger lengths by v,
        i.e.: l_new = l + v
        Empirically v should be in [0, 0.02]
        """
        finger_segment_list = []
        for element in self.robot_root.iter('body'):
            for finger_part in ["proximal", "middle", "distal"]:
                if finger_part in element.get('name'):
                    finger_segment_list.append(element)

        for finger_segment in finger_segment_list:
            finger_segment.attrib['pos'] = self._translate(finger_segment.attrib['pos'], 0, 0, v)

    def robot_change_joint_stiffness(self, v=80.0):
        """
        Changes the joint stiffness
        v -- stiffness value
        """
        palm = [i for i in self.robot_root.iter('body') if i.get('name')=='robot0:palm'][0]
        joints = [i for i in palm.iter('joint')]
        for i in joints:
            i.attrib['stiffness'] = str(v)

    # def robot_change_spring_default(self, a, b):
    #     """
    #     Changes default finger position by changing the default lengths for the springs in the proximal, middle and distal finger joints.
    #     All are in the range [0, 1.571], where 0 is for fully extended and 1.571 is for fully bent.
    #     a -- thumb
    #     b -- others
    #     """
    #     palm = [i for i in self.robot_root.iter('body') if i.get('name')=='robot0:palm'][0]
    #     joints = [i for i in palm.iter('joint')]
    #     for i in joints:
    #         # print(i.attrib['name'])
    #         if("TH" in i.attrib['name']):
    #             if(i.attrib['name'] == "robot0:THJ0"):
    #                 i.attrib['springref'] = str(-a)
    #             elif(i.attrib['name'] == "robot0:THJ3"):
    #                 i.attrib['springref'] = str(a)
    #         elif not ("J3" in i.attrib['name'] or "J4" in i.attrib['name']):
    #             i.attrib['springref'] = str(b)

    def robot_change_finger_spring_default(self, a=0.2):
            """
            Changes default finger position by changing the default lengths for the springs in the proximal, middle and distal finger joints.
            a is in the range [0, 1.571], where 0 is for fully extended and 1.571 is for fully bent.
            a -- default position for all three fingers (original: 0.2)
            """
            palm = [i for i in self.robot_root.iter('body') if i.get('name')=='robot0:palm'][0]
            for i in palm.iter('joint'):
                if not any(_ in i.attrib['name'] for _ in ["J4","J3","TH"]):
                    i.attrib['springref'] = str(a)
                    
    def robot_change_thumb_spring_default(self, a=0.1, b=1.6, c=-0.8):
            """
            Changes default thumb position by changing the default lengths for the thumb joints.

            a -- thumb base rotation outside of the palm plane(gripping), range [-1.047, 1.047], where 1.047 is for fully gripped
            b -- thumb base rotation in the palm plane, range [0, 1.6], where 0 is when thumb is closest to the index finger(adduction)
            c -- distal knuckle bending, range [-1.571, 0], where -1.571 is for fully bent
            """
            thbase = [i for i in self.robot_root.iter('body') if i.get('name')=='robot0:thbase'][0]
            for i in thbase.iter('joint'):
                if(i.attrib['name'] == "robot0:THJ4"):
                    i.attrib['springref'] = str(a)
                elif(i.attrib['name'] == "robot0:THJ3"):
                    i.attrib['springref'] = str(b)
                elif(i.attrib['name'] == "robot0:THJ0"):
                    i.attrib['springref'] = str(c)

    def robot_change_friction(self, a=1, b=0.005, c=0.0001):
        """
        Changes the friction in robot's fingertips
        a -- translational friction
        b -- rotational friction
        c -- rolling friction
        """
        for site in ['th','ff','mf','rf','lf']:
            body = [i for i in self.robot_root.iter('body') if i.get('name')=='robot0:'+site+'distal'][0]
            tip = [i for i in body.iter('geom') if i.get('name')=='robot0:C_'+site+'distal'][0]
            f = ' '.join(map(str, [a,b,c]))
            tip.set('friction',f)

    def debug(self):
        self.printer(self.lift_root[1])

    def printer(self, e):
        print("Curr: " + e.tag, e.attrib)
        print("Children: ")
        for child in e:
            print(child.tag, child.attrib)
        print()

    def export_XML(self):
        self.lift_tree.write(Parameterizer.modified_lift)
        self.robot_tree.write(Parameterizer.modified_robot)


pm = Parameterizer()
pm.randomize(4)
# pm.robot_change_finger_length(0.02)
# pm.robot_change_spring_default(1, 1)
# pm.robot_change_finger_spring_default(1.571)
# pm.robot_change_thumb_spring_default(1, 1.6, 0)
# pm.robot_change_joint_stiffness(5)
# pm.object_change_slope(0.025, 0.04, 0.12, 0.001)
# pm.translate_object(1, 1, 1)
pm.export_XML()
