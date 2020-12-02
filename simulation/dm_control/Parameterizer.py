import shutil, os
import copy

import numpy as np
import xml.etree.ElementTree as ET

class Parameterizer:
    unmodified_lift = os.getcwd() + "/environments/assets/passive_hand_unmodified/lift.xml"
    unmodified_robot = os.getcwd() + "/environments/assets/passive_hand_unmodified/robot.xml"

    modified_lift = os.getcwd() + "/environments/assets/passive_hand/lift.xml"
    modified_robot = os.getcwd() + "/environments/assets/passive_hand/robot.xml"

    def __init__(self):
        """Initialises an xml tree for lift.xml and robot.xml"""

        self.lift_tree = ET.parse(Parameterizer.unmodified_lift)
        self.robot_tree = ET.parse(Parameterizer.unmodified_robot)

        self.lift_root = self.lift_tree.getroot()
        self.robot_root = self.robot_tree.getroot()
    # def num_to_str(self, *args):
    #     return ' '.join(args)

    def _translate(self, pos: str, dx, dy, dz) -> str:
        """takes a position string, translates it by dxyz, and returns it"""

        xyz = np.array([float(x) for x in pos.split(' ')])
        dxyz = np.array([dx, dy, dz])
        xyz = list(xyz + dxyz)
        xyz = [str(x) for x in xyz]
        xyz = ' '.join(xyz)
        return xyz

    def object_translate(self, dx, dy, dz):
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
            self.printer(cylinder)

def robot_change_equilibrium_pos(self, a, b, c):
        """
        Changes the eq. pos
        a,b,c -- 
        """
        palm = [i for i in self.robot_root.iter('body') if i.get('name')=='robot0:palm'][0]
        joints = [i for i in palm.iter('joint')]
        for j in joints:
            j.attrib['springref'] = str(a)


    def robot_change_joint_stiffness(self, v):
        """
        Changes the joint stiffness
        v -- stiffness value
        """
        palm = [i for i in self.robot_root.iter('body') if i.get('name')=='robot0:palm'][0]
        joints = [i for i in palm.iter('joint')]
        for i in joints:
            i.attrib['stiffness'] = str(v)

    def robot_change_spring_default(self, a, b):
        """
        Changes default finger position by changing the default lengths for the springs in the proximal, middle and distal finger joints.
        All are in the range [0, 1.571], where 0 is for fully extended and 1.571 is for fully bent.
        a -- thumb
        b -- others
        """
        palm = [i for i in self.robot_root.iter('body') if i.get('name')=='robot0:palm'][0]
        joints = [i for i in palm.iter('joint')]
        for i in joints:
            # print(i.attrib['name'])
            if("TH" in i.attrib['name']):
                if(i.attrib['name'] == "robot0:THJ0"):
                    i.attrib['springref'] = str(-a)
                elif(i.attrib['name'] == "robot0:THJ3"):
                    i.attrib['springref'] = str(a)
            elif not ("J3" in i.attrib['name'] or "J4" in i.attrib['name']):
                i.attrib['springref'] = str(b)

    def robot_change_friction(self, a, b, c):
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

    def export_XML(self):
        self.lift_tree.write(Parameterizer.modified_lift)
        self.robot_tree.write(Parameterizer.modified_robot)


pm = Parameterizer()
pm.robot_change_spring_default(1, 1)
# pm.robot_change_joint_stiffness(5)
# pm.object_change_slope(0.025, 0.04, 0.12, 0.001)
# pm.translate_object(1, 1, 1)
pm.export_XML()
