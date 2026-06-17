"""Build the Go2 ``MjModel`` for the RMA MJX env.

Torque is computed in Python (PD on a residual joint target) and applied through
``data.qfrc_applied``, so per-env Kp/Kd/motor-strength randomization needs no
model edits and the action->torque path is identical in MJX training and the CPU
Controller. The model's own actuators are left in place but never driven.

Sensors are stripped: MJX supports only a subset, the env reads state straight
from ``mjx.Data``, and gym-quadruped recreates its own sensors at eval time.

Collisions are restricted to foot<->terrain. The Go2 leg/trunk primitives are
cylinders/boxes that MJX cannot narrow-phase against a plane or heightfield, and
an exploring policy penetrating them blows contact forces up to NaN. The grader's
"a non-foot body touching the ground is a fall" rule is instead reproduced by the
terrain-relative base-height termination in ``Go2Env``.

Requires mujoco >= 3.2 for the MjSpec editing API.
"""
from __future__ import annotations

import os

import mujoco

from .terrain import fractal_heightfield
from .go2_constants import LEGS_QPOS, resolve_model_path


def _foot_only_collisions(model: mujoco.MjModel) -> None:
    """Keep only foot<->terrain collision pairs, via contype/conaffinity masks.

    Feet collide with the world; the world (floor/heightfield) only acts as their
    target; every other robot geom is disabled. This avoids both MJX-unsupported
    narrow phases and the O(n^2) terrain<->terrain pairs that stall ``put_model``.
    """
    foot_names = set(LEGS_QPOS)
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if model.geom_bodyid[g] == 0:          # world / terrain
            model.geom_contype[g] = 1
            model.geom_conaffinity[g] = 0      # collide only as a foot target
        elif name in foot_names:               # foot
            model.geom_contype[g] = 1
            model.geom_conaffinity[g] = 1
        else:                                  # other robot geom
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0
    model.geom_margin[:] = 0.0                 # unused; zero for MJX safety
    model.geom_gap[:] = 0.0


def _strip_sensors(spec: "mujoco.MjSpec") -> None:
    for s in list(spec.sensors):
        try:
            spec.delete(s)
        except Exception:
            pass


def _has_floor(spec: "mujoco.MjSpec") -> bool:
    return any(g.type == mujoco.mjtGeom.mjGEOM_PLANE
               for g in spec.worldbody.geoms)


def _add_floor(spec: "mujoco.MjSpec") -> None:
    spec.worldbody.add_geom(name="rma_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0.0, 0.0, 0.05], pos=[0, 0, 0])


def _add_heightfield(spec: "mujoco.MjSpec", cfg) -> None:
    """Replace the floor with one shared fractal heightfield.

    A single hfield (1 geom) is the MJX-efficient way to train on uneven terrain;
    per-env terrain variety comes from randomized spawn positions, not per-env
    fields (the model is shared across the vmapped batch).
    """
    n, r = cfg.hfield_grid, cfg.hfield_radius
    hf = fractal_heightfield(size=n, octaves=cfg.fractal_octaves,
                             lacunarity=cfg.fractal_lacunarity, gain=cfg.fractal_gain,
                             base_frequency=cfg.fractal_base_freq, seed=0)
    spec.add_hfield(name="rma_terrain", size=[r, r, cfg.fractal_z_scale, 0.1],
                    nrow=n, ncol=n, userdata=hf.flatten().tolist())
    for g in list(spec.worldbody.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            spec.delete(g)
    spec.worldbody.add_geom(name="rma_terrain_geom", type=mujoco.mjtGeom.mjGEOM_HFIELD,
                            hfieldname="rma_terrain", pos=[0, 0, 0])


def build_model(cfg, model_path: str) -> mujoco.MjModel:
    # "auto"/None -> the Go2 bundled in gym-quadruped, so training and the grader
    # share the exact same model.
    model_path = resolve_model_path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. For the default (gym-quadruped "
            f"bundled Go2) just `pip install gym-quadruped`.")
    spec = mujoco.MjSpec.from_file(model_path)

    _strip_sensors(spec)
    if cfg.terrain == "hfield":
        _add_heightfield(spec, cfg)
    elif not _has_floor(spec):
        _add_floor(spec)                       # bare robot MJCF has no ground

    spec.option.timestep = cfg.physics_dt
    try:                                       # stable MJX contact: Newton solver
        spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
        spec.option.iterations = 4
        spec.option.ls_iterations = 8
    except Exception:
        pass

    model = spec.compile()
    _foot_only_collisions(model)
    return model
