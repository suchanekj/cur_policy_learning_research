from dm_control.rl import control
from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
import numpy as np

mjlib = mjbindings.mjlib
mjtObj = mjbindings.enums.mjtObj

class Physics(mujoco.Physics):
    def get_site_vel(self, site_name, is_local):
        #6DOF Vector with first 3 as velr and second 3 as velp
        vels = np.zeros(6)
        mjlib.mj_objectVelocity(self.model.ptr, self.data.ptr, mjtObj.mjOBJ_SITE, self.model.name2id(site_name, 'site'), vels, int(is_local))
        return vels