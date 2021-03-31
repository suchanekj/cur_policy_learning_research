import numpy as np
from dm_control.mujoco.wrapper import mjbindings

EQ_WELD = mjbindings.enums.mjtEq.mjEQ_WELD

def mocap_set_action(physics, action):
    """
    Sets action through a delta in rotation and position of mocap
    """
    if physics.model.nmocap > 0:
        action, _ = np.split(action, (physics.model.nmocap * 7,))
        action = action.reshape(physics.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(physics)
        physics.data.mocap_pos[:] = physics.data.mocap_pos + pos_delta
        physics.data.mocap_quat[:] = physics.data.mocap_quat + quat_delta

def reset_mocap_welds(physics):
    """
    Resets the mocap welds that we use for actuation.
    """
    if physics.model.nmocap > 0 and physics.model.eq_data is not None:
        for i in range(physics.model.eq_data.shape[0]):
            if physics.model.eq_type[i] == EQ_WELD:
                physics.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
    physics.forward()

def reset_mocap2body_xpos(physics):
    """
    Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.

    Essentially make sure that the position that the mocap is set to works with the rest of the joints and arms
    """

    if (physics.model.eq_type is None or
        physics.model.eq_obj1id is None or
        physics.model.eq_obj2id is None):
        return
    for eq_type, obj1_id, obj2_id in zip(physics.model.eq_type,
                                         physics.model.eq_obj1id,
                                         physics.model.eq_obj2id):
        if eq_type != EQ_WELD:
            continue

        mocap_id = physics.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = physics.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert (mocap_id != -1)
        physics.data.mocap_pos[mocap_id][:] = physics.data.xpos[body_idx]
        physics.data.mocap_quat[mocap_id][:] = physics.data.xquat[body_idx]