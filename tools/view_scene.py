"""Visualize the exact Go2 + terrain model used for RMA training.

Builds the model through the same code path as the trainer
(``rma.envs.build_model.build_model``), so what you see -- robot, injected
fractal heightfield, collision setup -- matches what the policy trains on.

Run on a machine with a display (your local PC), using the project venv:

    python tools/view_scene.py                 # default: terrain="hfield"
    python tools/view_scene.py --terrain scene # flat floor (gym-quadruped base)
    python tools/view_scene.py --model ../unitree_mujoco/unitree_robots/go2/scene_jagged.xml
    python tools/view_scene.py --z-scale 0.15  # exaggerate the heightfield

Controls: drag to orbit, right-drag to pan, scroll to zoom. Press the spacebar
to start/pause passive simulation (the robot will collapse since there is no
controller -- that's expected; this is just for inspecting the scene).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Make the repo root importable when this file is run directly (python
# tools/view_scene.py) -- otherwise only tools/ is on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
import mujoco.viewer

from rma.config import Config
from rma.envs.build_model import build_model
from rma.envs.go2_constants import resolve_layout, NOMINAL_POSE, INIT_BASE_HEIGHT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", choices=["hfield", "scene"], default=None,
                    help="override cfg.env.terrain (default: config value)")
    ap.add_argument("--model", type=str, default=None,
                    help="override model path (default: gym-quadruped Go2)")
    ap.add_argument("--z-scale", type=float, default=None,
                    help="override fractal heightfield z-scale (m)")
    ap.add_argument("--sim", action="store_true",
                    help="run passive physics (robot falls without a policy)")
    args = ap.parse_args()

    cfg = Config()
    if args.terrain is not None:
        cfg.env.terrain = args.terrain
    if args.z_scale is not None:
        cfg.env.fractal_z_scale = args.z_scale
    model_path = args.model if args.model is not None else cfg.model_path

    model = build_model(cfg.env, model_path)
    layout = resolve_layout(model)
    data = mujoco.MjData(model)

    # Place the Go2 in its nominal standing pose on top of the terrain.
    data.qpos[0:3] = [0.0, 0.0, INIT_BASE_HEIGHT]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[layout.joint_qpos_adr] = NOMINAL_POSE
    mujoco.mj_forward(model, data)

    print(f"terrain={cfg.env.terrain}  geoms={model.ngeom}  "
          f"z_scale={cfg.env.fractal_z_scale}  (sim={'on' if args.sim else 'off'})")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if args.sim:
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(cfg.env.physics_dt if args.sim else 0.02)


if __name__ == "__main__":
    main()
