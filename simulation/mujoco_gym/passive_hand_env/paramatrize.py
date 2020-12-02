import time
import shutil, os
import copy
import numpy as np
import xml.etree.ElementTree as ET

class Paramatrizer():

    def __init__(self, env):
        self.env = env
        self.model = env.sim.model
        self.body_names = env.sim.model.body_names

        self.default_param = [2, 
                             1.46177789, 0.74909766, 0.60,
                             0.02, 0.02, 0.02,
                           #  1, 0.005, 0.0001,
                           0,0,0,
                             0.1]

        self.model_path = os.path.join(os.path.dirname(__file__),'assets','passive_hand')
        self.temp_xml_path = os.path.join(os.path.dirname(__file__), 'assets', 'tmp')
        if not os.path.exists(self.temp_xml_path):
            shutil.copytree(model_path, self.temp_xml_path)

    def reset_with_param(self, param, re_render=False):
        '''param: mass[1], object position[3], object size[3], friction[3]'''
        if param==None: param=self.default_param 
        m=param[0]
        objpos=param[1:4]
        objsize=param[4:7]
        fric=param[7:10]
        finger_extra_lenth=param[10]

        #modifying assets/tmp/robot.xml 
        tree = ET.parse(os.path.join(self.model_path, 'robot.xml'))
        root = tree.getroot()
        for site in ['th','ff','mf','rf','lf']:
            # site = 
            body = [i for i in root.iter('body') if i.get('name')=='robot0:'+site+'distal'][0]
            tip = [i for i in body.iter('geom') if i.get('name')=='robot0:C_'+site+'distal'][0]
            # l=tip.get('pos')
            # if l==None: l = '0 0 0'
            # l = l.split()
            # l[2] = str(float(l[-1])+finger_extra_lenth)
            # l = ' '.join(l)
            # print(l)
            # tip.set('pos',l) #0 0 0.013 # 0 0 0.032

            f = ' '.join(map(str, fric))
            tip.set('friction',f)

        path = os.path.join(self.temp_xml_path, 'robot.xml')
        tree.write(path)

        #modifying assets/tmp/lift.xml 
        tree = ET.parse(os.path.join(self.model_path, 'lift.xml'))
        root = tree.getroot()

        obj = [i for i in root.iter('body') if i.get('name')=='object0'][0]
        objpos = ' '.join(map(str, objpos))
        obj.set('pos', objpos)
        geom = obj.find('geom')
        geom.set('mass', str(m))
        fric = ' '.join(map(str, fric))
        geom.set('friction', fric)
        site = obj.find('site')
        objsize = ' '.join(map(str, objsize))
        site.set('size', objsize)

        new_path = os.path.join(self.temp_xml_path, 'lift.xml')
        tree.write(new_path)

        self.env.reset(reload_model=True, new_model_path=new_path, re_render=re_render)
        

    def change_object_mass(self, mass):
        '''change mass of given objects'''
        index = self.model.body_names2id('object0')
        self.model.body_mass[index] = mass

    def change_object_pos(self, pos):
        '''change pos of given objects'''
        self.model.body_pos[-1][:] = pos

    # def change_finger_rolling_friction(self, fric):
    #     '''changes rolling friction coefficients of the finger tips'''

    # def change_finger_sliding_friction(self, fric):
    #     '''changes sliding friction coefficients of the finger tips'''
