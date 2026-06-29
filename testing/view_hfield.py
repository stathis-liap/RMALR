"""Autonomously visualize the trained policy on the training heightfield (MJX env).

gym-quadruped's only rough scene is a 0.56 m jagged perlin field -- far harsher
than training. To see the policy on the *actual* terrain it learned on, this
renders the MJX ``Go2Env`` itself (textured, gently rolling fractal heightfield)
and drives it with the trained networks, copying each MJX state into a CPU MuJoCo
viewer. The robot walks on its own: the env issues random velocity commands and
resamples them periodically. The camera follows it; close the window to quit.
(For keyboard teleop, use ``testing.manual``.)

Run with the GLFW backend:
    MUJOCO_GL=glfw python -m testing.view_hfield \
        --phase1 checkpoints/phase1_final.pkl --phase2 checkpoints/phase2_final.pkl
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp
import mujoco

from rma.config import Config
from rma.envs.go2_env import Go2Env
from rma.models import networks
from rma.utils import load_pytree


class HfieldRunner:
    """Runs the RMA deployment loop on a single MJX env and exposes its qpos.

    Velocity commands come from the env itself (sampled at reset, resampled per
    ``cmd_resample_prob``), so the robot walks autonomously.
    """

    def __init__(self, cfg, phase1, phase2, mode, seed):
        self.mode = mode
        self.env = Go2Env(cfg, "auto")
        self.edim = cfg.net.env_factor_dim
        enc, pol, _, adapt = networks.build_networks(cfg)
        p1 = load_pytree(phase1)
        phi = load_pytree(phase2) if mode == "rma" else None
        model = self.env.mjx_model

        self._reset = jax.jit(lambda rng: self.env.reset(model, rng))
        self._step = jax.jit(lambda st, a: self.env.step(model, st, a, 1.0))
        self._act = jax.jit(
            lambda st, z: pol.apply(p1["policy"], st.obs, st.prev_action, z)[0])
        self._mu = jax.jit(lambda e: enc.apply(p1["encoder"], e))
        self._phi = jax.jit(lambda st: adapt.apply(phi, st.history)) if phi else None
        # no_adapt: constant mu(e_nominal) for the default robot (not mu(0)).
        e_nominal = self.env._privileged(model, jnp.ones(12), jnp.zeros(()))
        self._z_nominal = self._mu(e_nominal)

        self.rng = jax.random.PRNGKey(seed)
        self._respawn()

    def _z(self):
        if self.mode == "rma":
            return self._phi(self.state)
        if self.mode == "expert":               # true e_t (incl. real terrain h)
            return self._mu(self.state.e)
        return self._z_nominal                  # no_adapt: mu(e_nominal)

    def _respawn(self):
        self.rng, k = jax.random.split(self.rng)
        self.state = self._reset(k)
        self.t = 0
        self.z_async = self._z()

    def step_once(self):
        if self.mode == "rma":
            if self.t % 10 == 0:                 # phi at ~10 Hz, as deployed
                self.z_async = self._phi(self.state)
            z = self.z_async
        else:
            z = self._z()
        action = self._act(self.state, z)
        self.state = self._step(self.state, action)
        self.t += 1
        if float(self.state.done) > 0.5:
            self._respawn()
        return np.asarray(self.state.data.qpos), np.asarray(self.state.data.qvel)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1", default="checkpoints/phase1_final.pkl")
    ap.add_argument("--phase2", default="checkpoints/phase2_final.pkl")
    ap.add_argument("--mode", default="rma", choices=["rma", "no_adapt", "expert"])
    ap.add_argument("--z-scale", type=float, default=0.12,
                    help="terrain height (m); 0.10 = exact training, raise to roughen")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import mujoco.viewer  # here so --help works headless

    cfg = Config()
    cfg.env.terrain = "hfield"
    cfg.env.fractal_z_scale = args.z_scale
    cfg.env.fractal_base_freq = 24          # shorter wavelength -> visibly rolling
    # Autonomous walk: keep command resampling on; drop DR churn and pushes so the
    # view is clean.
    cfg.env.resample_prob = cfg.env.push_prob = 0.0
    cfg.env.cmd_resample_prob = 0.01        # ~new command every ~100 steps (2 s)
    # Forgiving respawn: only reset on a real fall -- never on the episode timeout
    # (the robot keeps walking), nor on the gait's natural bob or a recoverable
    # tilt on the rolling terrain.
    cfg.env.episode_length = 10 ** 9
    cfg.env.min_base_height = 0.16
    cfg.env.upright_z_thresh = -0.3

    runner = HfieldRunner(cfg, args.phase1, args.phase2, args.mode, args.seed)
    mj_model = runner.env.mj_model
    mj_data = mujoco.MjData(mj_model)

    # Warm up before the window opens: the first step_once triggers JIT compilation
    # (a few seconds) and the robot drops the last cm onto the terrain. Doing it now
    # means the viewer opens with the robot already settled and walking, instead of
    # rendering the default at-origin pose (inside the ground) during the compile.
    print("compiling + settling (a few seconds)...", flush=True)
    for _ in range(25):
        qpos, qvel = runner.step_once()
    mj_data.qpos[:] = qpos
    mj_data.qvel[:] = qvel
    mujoco.mj_forward(mj_model, mj_data)

    print("autonomous hfield viewer -- random commands; close the window to quit")
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = runner.env.layout.trunk_body_id
        viewer.cam.distance, viewer.cam.elevation = 2.5, -20.0
        dt, frame, t0 = runner.env.dt, 0, time.time()
        while viewer.is_running():
            qpos, qvel = runner.step_once()
            mj_data.qpos[:] = qpos
            mj_data.qvel[:] = qvel
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            frame += 1
            if frame % 25 == 0:                # show the current command
                c = np.asarray(runner.state.command)
                print(f"\rcmd  vx={c[0]:+.2f}  vy={c[1]:+.2f}  yaw={c[2]:+.2f}   ",
                      end="", flush=True)
            lag = frame * dt - (time.time() - t0)  # pace to ~real time
            if lag > 0:
                time.sleep(min(lag, 0.05))
    print()


if __name__ == "__main__":
    main()
