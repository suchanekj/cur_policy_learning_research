import time
import shutil, os
import copy
import numpy as np
import xml.etree.ElementTree as ET

class Paramatrizer():

    def __init__(self, env):
        self.model = env.sim.model
        self.body_names = env.sim.model.body_names

        self.temp_xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'tmp')


    def change_mass(self, body_names, mode='gaussian') -> np.array:
        '''change and return mass of given objects'''

        for body in body_names:
            index = self.model.body_names2id(body)
            if mode == 'gaussian':
                m = self.model.body_mass[index]
                self.model.body_mass[index] = np.max(0, np.random.randn(m, m/3))
            elif mode == 'uniform':
                self.model.body_mass[index] = np.random.uniform(0, 2*self.model.body_mass[index])

        changes = copy.deepcopy(self.model.body_mass)
        return changes

    def change_object_pos(self) -> np.array:
        '''change and return pos of given objects'''
        
        self.model.body_pos[-1][0] = np.random.uniform(1.2, 1.55)
        self.model.body_pos[-1][1] = np.random.uniform(0.4, 1.1)

        changes = copy.deepcopy(self.model.body_pos)
        return changes


    def create_xml(self, changes) -> str:
        '''create and return path to new xml model'''

        model_path = os.path.join(os.path.dirname(__file__),'assets','passive_hand')
        if not os.path.exists(self.temp_xml_path):
            shutil.copytree(model_path, self.temp_xml_path)

        tree = ET.parse(os.path.join(model_path, 'lift.xml'))
        root = tree.getroot()
        wb = [i for i in root.iter('body') if i.get('name')=='object0'][0]  # change cylinder position
        pos = ['1.46177789', '0.74909766', '0.60']
        pos[1] = str(np.random.uniform(0.3, 1.1))
        pos = ' '.join(pos)
        wb.set('pos', pos)
        new_path = os.path.join(self.temp_xml_path, 'lift.xml')
        tree.write(new_path)

        return new_path