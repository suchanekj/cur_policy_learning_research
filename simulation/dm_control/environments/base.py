from dm_control.rl import control
from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
import numpy as np

mjlib = mjbindings.mjlib
mjtObj = mjbindings.enums.mjtObj

class Physics(mujoco.Physics):
    def get_site_vel(self, site_name, is_local):
        #6DOF Vector with first 3 as velr and second 3 as velp
        vel = np.zeros(6)
        mjlib.mj_objectVelocity(self.model.ptr, self.data.ptr, mjtObj.mjOBJ_SITE, self.model.name2id(site_name, 'site'), vel, int(is_local))
        return vel

    def get_geom_vel(self, geom_name, is_local):
        vel = np.zeros(6)
        mjlib.mj_objectVelocity(self.model.ptr, self.data.ptr, mjtObj.mjOBJ_GEOM, self.model.name2id(geom_name, 'geom'), vel, int(is_local))
        return vel

    def get_actuators_from_site_pos(self, site_name):
        id = self.model.name2id(site_name, 'site')
        jacp = np.zeros(3 * self.model.nv)
        jacr = np.zeros(3 * self.model.nv)
        mjlib.mj_jacSite(self.model.ptr, self.data.ptr, jacp, jacr, id)
        mjlib.mj_inverse(self.model.ptr, self.data.ptr)
        print(self.data.xfrc_applied.shape)
        print(jacp.shape)
        print(self.data.qfrc_inverse.shape)
