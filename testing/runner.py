"""Build a stressed gym-quadruped env and roll the Controller through it.

Importing ``rma.eval_gym`` here applies the project's IMU-init patch and the
Project-3 reward, and gives us the proprioceptive observation set -- so the
Controller sees exactly what it does at grading time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import mujoco

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.imu import IMU

import rma.eval_gym as eval_gym          # applies IMU patch + reward; PROPRIOCEPTIVE_OBS
from .scenarios import Scenario

_IMU_KW = {"accel_name": "imu_acc", "gyro_name": "imu_gyro", "imu_site_name": "imu"}
_TRUNK_NAMES = ("base", "base_link", "trunk")


@dataclass
class EpisodeResult:
    survived: bool
    steps: int
    lin_err: float
    yaw_err: float


def build_env(sc: Scenario) -> QuadrupedEnv:
    """gym-quadruped env with the scenario's terrain/friction/payload/COM applied."""
    env = QuadrupedEnv(
        robot="go2", scene=sc.scene,
        base_vel_command_type="random_rotate_reset",
        ref_base_lin_vel=(0.0, 1.0), ref_base_ang_vel=(-1.0, 1.0),
        ground_friction_coeff=tuple(sc.friction),
        state_obs_names=eval_gym.PROPRIOCEPTIVE_OBS,
        sensors=(IMU,), sensors_kwargs=(_IMU_KW,),
    )
    if sc.payload or any(sc.com_shift):
        for name in _TRUNK_NAMES:
            bid = mujoco.mj_name2id(env.mjModel, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                env.mjModel.body_mass[bid] += float(sc.payload)
                env.mjModel.body_ipos[bid, 0] += float(sc.com_shift[0])
                env.mjModel.body_ipos[bid, 1] += float(sc.com_shift[1])
                break
    return env


def _privileged_from_env(env: QuadrupedEnv, sc: Scenario) -> np.ndarray:
    """The sim's true e_t (17-dim) for expert mode, read after reset.

    Layout matches training (go2_env._privileged): payload(1) + com_xy(2) +
    motor_strength(12) + friction(1) + terrain_height(1). gym-quadruped sets the
    foot geom friction to the per-episode sampled value, so it reads back exactly
    as training built it; terrain_height is left 0 (flat).
    """
    m = env.mjModel
    bid = -1
    for n in _TRUNK_NAMES:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        if bid >= 0:
            break
    mass = float(m.body_mass[bid])
    com = np.asarray(m.body_ipos[bid, :2], dtype=np.float32)
    fgid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "FL")
    friction = float(m.geom_friction[fgid, 0]) if fgid >= 0 else float(sc.friction[0])
    motor = sc.motor_scale * np.ones(12, dtype=np.float32)
    return np.concatenate([[mass], com, motor, [friction], [0.0]]).astype(np.float32)


def _apply_push(env: QuadrupedEnv, sc: Scenario, rng) -> None:
    """Instantaneous base-velocity shove (added momentum, world frame)."""
    kick = np.zeros(3)
    if "x" in sc.push_axes:
        kick[0] = rng.choice([-1.0, 1.0]) * sc.push_vel
    if "y" in sc.push_axes:
        kick[1] = rng.choice([-1.0, 1.0]) * sc.push_vel
    env.mjData.qvel[0:3] += kick


def run_scenario(sc, ctrl, episodes, max_steps, seed_base, render=False):
    """Run one scenario for `episodes` and return a list of EpisodeResult."""
    env = build_env(sc)
    results = []
    for ep in range(episodes):
        obs = env.reset(seed=seed_base + ep)
        ctrl.reset(seed=seed_base + ep)
        if ctrl.mode == "expert":          # oracle: feed the true env factors
            ctrl.set_privileged(_privileged_from_env(env, sc))
        push_rng = np.random.default_rng(seed_base + ep)
        term = trunc = False
        steps, lin, yaw = 0, [], []
        t_wall = time.time()
        while not (term or trunc) and steps < max_steps:
            action, _ = ctrl.act(obs)
            if sc.motor_scale != 1.0:
                action = action * sc.motor_scale
            obs, _, term, trunc, _ = env.step(action)
            steps += 1
            lin.append(np.linalg.norm(obs["base_lin_vel_err:base"][:2]))
            yaw.append(abs(obs["base_ang_vel_err:base"][2]))
            if sc.push_interval and steps % sc.push_interval == 0:
                _apply_push(env, sc, push_rng)
            if render and steps % 10 == 0:
                env.render()
                lag = steps * 0.002 - (time.time() - t_wall)
                if lag > 0:
                    time.sleep(min(lag, 0.05))
        results.append(EpisodeResult(not term, steps,
                                     float(np.mean(lin)), float(np.mean(yaw))))
        print(f"  [ep {ep:02d}] steps={steps:4d} lin_err={np.mean(lin):.3f} "
              f"yaw_err={np.mean(yaw):.3f} {'survived' if not term else 'FELL'}")
    env.close()
    return results


def summarize(name, mode, results):
    """Aggregate per-episode results into one printable row dict."""
    n = len(results)
    return {
        "scenario": name,
        "mode": mode,
        "survival": 100.0 * sum(r.survived for r in results) / n,
        "steps": np.mean([r.steps for r in results]),
        "lin_err": np.mean([r.lin_err for r in results]),
        "yaw_err": np.mean([r.yaw_err for r in results]),
    }
