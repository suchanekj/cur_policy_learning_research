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

    def _translate(self, pos: str, dx, dy, dz) -> str:
        """takes a position string, translates it by dxyz, and returns it"""

        xyz = np.array([float(x) for x in pos.split(' ')])
        dxyz = np.array([dx, dy, dz])
        xyz = list(xyz + dxyz)
        xyz = [str(x) for x in xyz]
        xyz = ' '.join(xyz)
        return xyz

    def modify_object(self, dx, dy, dz):
        """modifies lift_root by shifting the object position by xyz"""

        # needs to be updated once dhilan adds in his objects

        object = self.lift_root[4][4]
        pos = object.attrib['pos']
        object.attrib['pos'] = self._translate(pos, dx, dy, dz)

    # def modify_fingers(self, dx, dy=0, dz=0):


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
# pm.modify_object(1, 1, 1)
pm.export_XML()
# pm = Parameterizer()
