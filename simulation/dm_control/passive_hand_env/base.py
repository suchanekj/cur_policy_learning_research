from dm_control.rl import control
from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
import numpy as np

mjlib = mjbindings.mjlib

class Physics(mujoco.Physics):
    def site_jacp(self, site_name):
        site_id = self.model.name2id(site_name, 'site')
        jacp = np.zeros(3 * self.model.nv)
        mjlib.mj_jacSite(self.model.ptr, self.data.ptr, jacp, None, site_id)
        return jacp

    def site_jacr(self, site_name):
        site_id = self.model.name2id(site_name, 'site')
        jacr = np.zeros(3 * self.model.nv)
        mjlib.mj_jacSite(self.model.ptr, self.data.ptr, None, jacr, site_id)
        return jacr

    def site_xvelp(self, site_name):
        jacp = self.site_jacp(site_name).reshape((3, self.model.nv))
        return np.dot(jacp, self.data.qvel)

    def site_xvelr(self, site_name):
        jacr = self.site_jacr(site_name).reshape((3, self.model.nv))
        return np.dot(jacr, self.data.qvel)

