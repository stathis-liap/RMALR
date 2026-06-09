"""Robot-specific name/index resolution for the Unitree Go2 MuJoCo model.

Joint ordering is the crux of train<->deploy consistency. In the Go2 MJCF the
generalized coordinates are laid out as:

    qpos = [base_pos(3), base_quat_wxyz(4), 12 hinge joints]
    qvel = [base_lin_vel(3, world), base_ang_vel(3, body), 12 hinge dofs]

The 12 hinge joints appear in **FL, FR, RL, RR** order (per leg: hip, thigh,
calf). This is exactly the order gym-quadruped returns as ``qpos_js`` /
``qvel_js``. We therefore build the policy state and the PD torque in this
*qpos order* throughout, and only permute to the actuator/``ctrl`` order
(FR, FL, RR, RL) at the very end in the deployment Controller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import mujoco


# qpos / qvel joint order.
LEGS_QPOS = ("FL", "FR", "RL", "RR")
# Actuator / ctrl order (what gym-quadruped's action space expects).
LEGS_CTRL = ("FR", "FL", "RR", "RL")
JOINT_SUFFIXES = ("hip", "thigh", "calf")

# Nominal standing pose (rad), per leg [hip, thigh, calf]. Go2 keyframe "home".
NOMINAL_POSE = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float32)

# Per-joint torque limit (Nm), qpos order. Go2 motors: hip/thigh 23.7, calf 45.43.
TORQUE_LIMIT = np.array([23.7, 23.7, 45.43] * 4, dtype=np.float32)

# Initial base height (m) for resets (Go2 stands ~0.30-0.33 m).
INIT_BASE_HEIGHT = 0.32

TRUNK_BODY_NAME = "base_link"


@dataclass
class Go2Layout:
    trunk_body_id: int
    joint_qpos_adr: np.ndarray   # (12,) qpos address of each hinge (qpos order)
    joint_qvel_adr: np.ndarray   # (12,) qvel/dof address of each hinge
    foot_body_ids: List[int]     # 4 foot body ids (qpos leg order)
    foot_geom_ids: List[int]     # 4 foot collision geom ids (qpos leg order)


def _name2id(model, objtype, name) -> int:
    return mujoco.mj_name2id(model, objtype, name)


def resolve_layout(model: mujoco.MjModel) -> Go2Layout:
    trunk_id = _name2id(model, mujoco.mjtObj.mjOBJ_BODY, TRUNK_BODY_NAME)
    assert trunk_id >= 0, f"could not find trunk body '{TRUNK_BODY_NAME}'"

    qpos_adr, qvel_adr = [], []
    for leg in LEGS_QPOS:
        for suf in JOINT_SUFFIXES:
            jid = _name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{suf}_joint")
            assert jid >= 0, f"missing joint {leg}_{suf}_joint"
            qpos_adr.append(model.jnt_qposadr[jid])
            qvel_adr.append(model.jnt_dofadr[jid])

    foot_body_ids, foot_geom_ids = [], []
    for leg in LEGS_QPOS:
        bid = _name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot")
        assert bid >= 0, f"missing foot body {leg}_foot"
        foot_body_ids.append(bid)
        # Foot collision geom is named exactly after the leg ("FL", "FR", ...).
        gid = _name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg)
        assert gid >= 0, f"missing foot geom '{leg}'"
        foot_geom_ids.append(gid)

    return Go2Layout(
        trunk_body_id=trunk_id,
        joint_qpos_adr=np.asarray(qpos_adr, dtype=np.int32),
        joint_qvel_adr=np.asarray(qvel_adr, dtype=np.int32),
        foot_body_ids=foot_body_ids,
        foot_geom_ids=foot_geom_ids,
    )


def ctrl_from_qpos_permutation(model: mujoco.MjModel) -> np.ndarray:
    """Index array mapping a qpos-ordered (12,) joint vector to ctrl order.

    ``action_ctrl = tau_qpos[perm]`` reorders a torque vector computed in qpos
    order (FL, FR, RL, RR) into the actuator order gym-quadruped applies.
    """
    # qpos order joint ids in sequence
    qpos_joint_names = [f"{leg}_{suf}_joint"
                        for leg in LEGS_QPOS for suf in JOINT_SUFFIXES]
    qpos_index = {name: i for i, name in enumerate(qpos_joint_names)}

    perm = np.zeros(model.nu, dtype=np.int32)
    for a in range(model.nu):
        jid = model.actuator_trnid[a, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        perm[a] = qpos_index[jname]
    return perm
