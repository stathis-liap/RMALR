"""Build the Go2 MjModel used by the RMA MJX env.

We load the Go2 scene (e.g. the jagged scene from tools/make_jagged_scene.py).
RMA applies joint torques computed in Python (PD on a residual joint-position
target) directly via ``data.qfrc_applied``, which lets us randomize
Kp/Kd/motor-strength per environment without editing the model and keeps the
action -> torque path identical between MJX training and the CPU deployment
Controller. The model's own torque actuators are left in place but never driven
(``data.ctrl`` stays 0, so they add zero generalized force); removing them would
invalidate the Go2 "home" keyframe's ctrl vector.

We *do* strip <sensor> definitions: MJX only supports a subset of sensor types,
and the RMA env reads everything it needs (joint state, base pose/velocity,
contacts) directly from mjx.Data. The benchmark's own sensors (IMU etc.) are
re-created by gym-quadruped at evaluation time.

Requires mujoco >= 3.2 for the MjSpec editing API.
"""
from __future__ import annotations

import os

import mujoco

from .terrain import fractal_heightfield
from .go2_constants import LEGS_QPOS


def _foot_only_collisions(model: mujoco.MjModel) -> None:
    """Restrict collisions to foot <-> terrain only (critical for MJX).

    Two problems with the stock Unitree Go2 + jagged scene under MJX:

    1. The Go2 MJCF carries box/cylinder collision primitives on the trunk and
       legs; MJX does not implement some of those narrow-phase pairs (e.g.
       cylinder<->box), and self/body collisions are irrelevant for locomotion.
    2. The jagged scene has ~700 terrain boxes, all with the default
       contype/conaffinity=1. MJX enumerates collision pairs host-side at
       ``put_model``, so leaving them mutually collidable creates ~C(700,2)
       box<->box pairs and makes ``put_model`` hang / run out of memory.

    Collision policy applied here (via contype/conaffinity bitmasks):
      * feet (sphere geoms named FL/FR/RL/RR): contype=1, conaffinity=1
      * world/terrain geoms (floor plane + boxes): conaffinity=0 (they act only
        as a contact *target* for the feet -> no terrain<->terrain pairs)
      * all other robot geoms: collisions disabled entirely
    Result: only foot<->terrain (and trivial foot<->foot) pairs survive.
    Termination uses base height/orientation, not base contact, so disabling
    trunk collision in MJX is safe.
    """
    foot_names = set(LEGS_QPOS)
    for g in range(model.ngeom):
        body_id = model.geom_bodyid[g]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if body_id == 0:                       # world / terrain geom
            model.geom_conaffinity[g] = 0      # collide only as feet's target
            model.geom_contype[g] = 1
        elif name in foot_names:               # foot
            model.geom_contype[g] = 1
            model.geom_conaffinity[g] = 1
        else:                                  # other robot geom
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0
    # MJX has not implemented margin/gap for some narrow-phase pairs (e.g.
    # hfield<->sphere). We don't use collision margins, so zero them globally.
    model.geom_margin[:] = 0.0
    model.geom_gap[:] = 0.0


def _strip_sensors(spec: "mujoco.MjSpec") -> None:
    for s in list(spec.sensors):
        try:
            spec.delete(s)
        except Exception:
            pass


def _add_heightfield(spec: "mujoco.MjSpec", cfg) -> None:
    """Best-effort: add a shared fractal heightfield and replace the floor.

    Caveat: MJX hfield collision support depends on your mujoco version. If this
    raises on your stack, keep `env.terrain="scene"`.
    """
    size = 256
    hf = fractal_heightfield(
        size=size,
        octaves=cfg.fractal_octaves,
        lacunarity=cfg.fractal_lacunarity,
        gain=cfg.fractal_gain,
        z_scale=cfg.fractal_z_scale,
        seed=0,
    )
    spec.add_hfield(
        name="rma_terrain",
        size=[10.0, 10.0, cfg.fractal_z_scale, 0.1],
        nrow=size,
        ncol=size,
        userdata=hf.flatten().tolist(),
    )
    for g in list(spec.worldbody.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            spec.delete(g)
    spec.worldbody.add_geom(
        name="rma_terrain_geom",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname="rma_terrain",
        pos=[0, 0, 0],
    )


def build_model(cfg, model_path: str) -> mujoco.MjModel:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run scripts/download_assets.sh first."
        )
    spec = mujoco.MjSpec.from_file(model_path)

    _strip_sensors(spec)

    if cfg.terrain == "hfield":
        _add_heightfield(spec, cfg)

    # Stable physics for MJX: Newton solver, small timestep.
    spec.option.timestep = cfg.physics_dt
    try:
        spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
        spec.option.iterations = 4
        spec.option.ls_iterations = 8
    except Exception:
        pass

    model = spec.compile()
    _foot_only_collisions(model)
    return model
