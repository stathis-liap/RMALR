"""MJX environment for RMA goal-conditioned velocity tracking on the Unitree Go2.

Target benchmark: gym-quadruped (Project 3). This env mirrors the benchmark's
proprioceptive observation and the Project-3 reward so the policy trained here
transfers to the ``Controller`` running inside gym-quadruped.

Key design notes
----------------
* Actuators are stripped from the model; torque is computed in Python (PD on a
  residual joint-position target) and applied via ``data.qfrc_applied``. The
  same PD map is reproduced in the deployment Controller, so the action->torque
  path is identical in MJX training and CPU evaluation. Go2 has direct-drive
  torque actuators, so the Controller simply returns this torque as ``ctrl``.
* Everything joint-indexed is in **qpos order** (FL, FR, RL, RR) -- identical to
  gym-quadruped's ``qpos_js`` / ``qvel_js``.
* The policy state ``x_t`` (36) =
      gravity_base(3) | base_ang_vel(3) | qpos_js(12) | qvel_js(12)
      | base_lin_vel_err(3) | base_ang_vel_err(3)
  i.e. only quantities the benchmark exposes, including the velocity command via
  the two error terms (goal conditioning).
* Privileged ``e_t`` (17) = payload(1), com(2), motor_strength(12), friction(1),
  terrain_height(1) -- fed only to the encoder/critic during Phase 1.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from flax import struct

from .go2_constants import (
    resolve_layout, NOMINAL_POSE, TORQUE_LIMIT, INIT_BASE_HEIGHT,
)
from .build_model import build_model


# Observation slice layout (see module docstring). Handy for downstream code.
OBS_SLICES = {
    "gravity": slice(0, 3),
    "ang_vel": slice(3, 6),
    "qpos_js": slice(6, 18),
    "qvel_js": slice(18, 30),
    "lin_vel_err": slice(30, 33),
    "ang_vel_err": slice(33, 36),
}


@struct.dataclass
class State:
    data: Any                  # mjx.Data
    obs: jnp.ndarray           # (36,) current state x_t
    history: jnp.ndarray       # (history_len, 36+12) of [x, a] pairs
    e: jnp.ndarray             # (17,) privileged env factors
    command: jnp.ndarray       # (3,) [vx, vy, yaw_rate] target (base frame)
    prev_action: jnp.ndarray   # (12,)
    prev_torque: jnp.ndarray   # (12,)
    # state-level randomization params
    kp: jnp.ndarray            # scalar
    kd: jnp.ndarray            # scalar
    motor_strength: jnp.ndarray  # (12,)
    # bookkeeping
    step: jnp.ndarray          # scalar int
    reward: jnp.ndarray        # scalar
    done: jnp.ndarray          # scalar
    # diagnostics: [tracking_lin, tracking_yaw, penalty_magnitude]
    metrics: jnp.ndarray       # (3,)
    rng: jnp.ndarray


def _quat_rotate_inverse(q, v):
    """Rotate world-frame vector v into the body frame given body quat (wxyz)."""
    w, u = q[0], q[1:4]
    t = 2.0 * jnp.cross(u, v)
    return v - w * t + jnp.cross(u, t)


def _rp_to_quat(roll, pitch):
    """Quaternion (wxyz) for a roll-then-pitch tilt (yaw-free)."""
    cr, sr = jnp.cos(roll / 2), jnp.sin(roll / 2)
    cp, sp = jnp.cos(pitch / 2), jnp.sin(pitch / 2)
    # q = qx(roll) * qy(pitch), Hamilton product
    return jnp.array([cr * cp, sr * cp, cr * sp, sr * sp])


def _sample_command(key, ecfg):
    k1, k2, k3 = jax.random.split(key, 3)
    vx = jax.random.uniform(k1, (), minval=ecfg.cmd_vx_range[0],
                            maxval=ecfg.cmd_vx_range[1])
    vy = jax.random.uniform(k2, (), minval=ecfg.cmd_vy_range[0],
                            maxval=ecfg.cmd_vy_range[1])
    wz = jax.random.uniform(k3, (), minval=ecfg.cmd_wz_range[0],
                            maxval=ecfg.cmd_wz_range[1])
    return jnp.stack([vx, vy, wz])


class Go2Env:
    """Stateless-by-design MJX environment (all state lives in `State`)."""

    def __init__(self, cfg, model_path: str):
        self.cfg = cfg
        self.rcfg = cfg.reward
        self.ecfg = cfg.env
        self.mj_model = build_model(self.ecfg, model_path)
        self.layout = resolve_layout(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model)

        self.dt = self.ecfg.physics_dt * self.ecfg.control_decimation
        self.nominal_pose = jnp.asarray(NOMINAL_POSE)
        self.torque_limit = jnp.asarray(TORQUE_LIMIT)
        self.qpos_adr = jnp.asarray(self.layout.joint_qpos_adr)
        self.qvel_adr = jnp.asarray(self.layout.joint_qvel_adr)
        self.foot_geom_ids = jnp.asarray(self.layout.foot_geom_ids)

        # cached init qpos/qvel. Spawn above the highest terrain point so feet
        # never start inside the heightfield (it raises ground by up to z_scale);
        # the drop-in also matches gym-quadruped's lift-until-clear reset.
        terrain_clear = (self.ecfg.fractal_z_scale
                         if self.ecfg.terrain == "hfield" else 0.0)
        self.spawn_z = INIT_BASE_HEIGHT + terrain_clear + self.ecfg.reset_drop_height
        q0 = np.array(self.mj_model.qpos0, dtype=np.float32).copy()
        q0[0:3] = [0.0, 0.0, self.spawn_z]
        q0[3:7] = [1.0, 0.0, 0.0, 0.0]
        q0[self.layout.joint_qpos_adr] = NOMINAL_POSE
        self.init_qpos = jnp.asarray(q0)
        self.init_qvel = jnp.zeros(self.mj_model.nv)

    # ------------------------------------------------------------------ utils
    def _kinematics(self, data):
        """Return the proprioceptive quantities used by obs and reward."""
        quat = data.qpos[3:7]
        gravity_b = _quat_rotate_inverse(quat, jnp.array([0.0, 0.0, -1.0]))
        lin_vel_b = _quat_rotate_inverse(quat, data.qvel[0:3])  # world->base
        ang_vel_b = data.qvel[3:6]                              # already body frame
        q = data.qpos[self.qpos_adr]
        qd = data.qvel[self.qvel_adr]
        return gravity_b, lin_vel_b, ang_vel_b, q, qd

    def _make_obs(self, gravity_b, lin_vel_b, ang_vel_b, q, qd, command):
        target_lin = jnp.array([command[0], command[1], 0.0])
        target_ang = jnp.array([0.0, 0.0, command[2]])
        lin_err = target_lin - lin_vel_b
        ang_err = target_ang - ang_vel_b
        return jnp.concatenate([gravity_b, ang_vel_b, q, qd, lin_err, ang_err])

    def _privileged(self, model, motor_strength):
        friction = model.geom_friction[self.foot_geom_ids[0], 0]
        payload = model.body_mass[self.layout.trunk_body_id]
        com = model.body_ipos[self.layout.trunk_body_id, 0:2]
        terrain_h = jnp.zeros(())
        return jnp.concatenate([
            payload.reshape(1), com, motor_strength,
            friction.reshape(1), terrain_h.reshape(1),
        ])

    # ----------------------------------------------------------------- reset
    def reset(self, model, rng):
        rng, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng, 8)
        ecfg = self.ecfg

        # Randomized initial state (mirrors gym-quadruped's reset, which the
        # benchmark applies at evaluation: joint pose/vel noise + base tilt).
        jp = jax.random.uniform(k5, (12,), minval=-ecfg.reset_joint_pos_noise,
                                maxval=ecfg.reset_joint_pos_noise)
        jv = jax.random.uniform(k6, (12,), minval=-ecfg.reset_joint_vel_noise,
                                maxval=ecfg.reset_joint_vel_noise)
        rp = jax.random.uniform(k7, (2,), minval=-ecfg.reset_rp_noise,
                                maxval=ecfg.reset_rp_noise)

        qpos = self.init_qpos
        qpos = qpos.at[self.qpos_adr].add(jp)
        qpos = qpos.at[3:7].set(_rp_to_quat(rp[0], rp[1]))
        qvel = self.init_qvel.at[self.qvel_adr].add(jv)

        data = mjx.make_data(model)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(model, data)

        kp = jax.random.uniform(k1, (), minval=self.ecfg.kp_range[0],
                                maxval=self.ecfg.kp_range[1])
        kd = jax.random.uniform(k2, (), minval=self.ecfg.kd_range[0],
                                maxval=self.ecfg.kd_range[1])
        ms = jax.random.uniform(k3, (self.cfg.net.action_dim,),
                                minval=self.ecfg.motor_strength_range[0],
                                maxval=self.ecfg.motor_strength_range[1])
        command = _sample_command(k4, self.ecfg)

        gravity_b, lin_vel_b, ang_vel_b, q, qd = self._kinematics(data)
        obs = self._make_obs(gravity_b, lin_vel_b, ang_vel_b, q, qd, command)
        e = self._privileged(model, ms)

        zero_a = jnp.zeros(self.cfg.net.action_dim)
        hist_row = jnp.concatenate([obs, zero_a])
        history = jnp.tile(hist_row, (self.ecfg.history_len, 1))

        return State(
            data=data, obs=obs, history=history, e=e, command=command,
            prev_action=zero_a, prev_torque=jnp.zeros(self.cfg.net.action_dim),
            kp=kp, kd=kd, motor_strength=ms,
            step=jnp.zeros((), jnp.int32), reward=jnp.zeros(()),
            done=jnp.zeros(()), metrics=jnp.zeros(3), rng=rng,
        )

    # ------------------------------------------------------------------ step
    def step(self, model, state: State, action, penalty_scale):
        rng, k_resample, k_dr, k_cmd, k_push = jax.random.split(state.rng, 5)

        # --- stochastic within-episode resample of state-level DR params ----
        do_resample = jax.random.uniform(k_resample) < self.ecfg.resample_prob
        kr1, kr2, kr3 = jax.random.split(k_dr, 3)
        new_kp = jax.random.uniform(kr1, (), minval=self.ecfg.kp_range[0],
                                    maxval=self.ecfg.kp_range[1])
        new_kd = jax.random.uniform(kr2, (), minval=self.ecfg.kd_range[0],
                                    maxval=self.ecfg.kd_range[1])
        new_ms = jax.random.uniform(kr3, (self.cfg.net.action_dim,),
                                    minval=self.ecfg.motor_strength_range[0],
                                    maxval=self.ecfg.motor_strength_range[1])
        kp = jnp.where(do_resample, new_kp, state.kp)
        kd = jnp.where(do_resample, new_kd, state.kd)
        motor_strength = jnp.where(do_resample, new_ms, state.motor_strength)

        # --- within-episode command resample (teaches command transitions) --
        do_cmd = jax.random.uniform(k_cmd) < self.ecfg.cmd_resample_prob
        kc, _ = jax.random.split(k_cmd)
        command = jnp.where(do_cmd, _sample_command(kc, self.ecfg), state.command)

        # --- optional external push (robustness to hidden-test disturbances) -
        kp_when, kp_dir = jax.random.split(k_push)
        do_push = jax.random.uniform(kp_when) < self.ecfg.push_prob
        push_dir = jax.random.normal(kp_dir, (3,))
        push_dir = push_dir.at[2].set(0.0)
        push_dv = jnp.where(do_push, self.ecfg.push_lin_vel, 0.0) * push_dir

        # --- PD torque, applied via qfrc_applied over control_decimation -----
        target_q = self.nominal_pose + self.ecfg.action_scale * action

        def pd_torque(data):
            q = data.qpos[self.qpos_adr]
            qd = data.qvel[self.qvel_adr]
            tau = motor_strength * (kp * (target_q - q) - kd * qd)
            return jnp.clip(tau, -self.torque_limit, self.torque_limit)

        # Apply the push as an instantaneous base-velocity perturbation.
        qvel0 = state.data.qvel.at[0:3].add(push_dv)
        data = state.data.replace(qvel=qvel0)

        def physics_substep(data, _):
            tau = pd_torque(data)
            qfrc = data.qfrc_applied.at[self.qvel_adr].set(tau)
            data = data.replace(qfrc_applied=qfrc)
            data = mjx.step(model, data)
            return data, tau

        data, taus = jax.lax.scan(physics_substep, data, None,
                                  length=self.ecfg.control_decimation)
        torque = taus[-1]

        gravity_b, lin_vel_b, ang_vel_b, q, qd = self._kinematics(data)
        obs = self._make_obs(gravity_b, lin_vel_b, ang_vel_b, q, qd, command)

        # --- reward (Project-3 _compute_reward) -----------------------------
        lin_err = obs[OBS_SLICES["lin_vel_err"]]
        ang_err = obs[OBS_SLICES["ang_vel_err"]]
        r = self.rcfg
        # Train on a gentler tracking sigma than the (very peaked) grading reward
        # so the policy gets a usable gradient toward the command. eval_gym.py
        # reports the true sigma=0.05 metric.
        tracking_lin = jnp.exp(-jnp.sum(lin_err[:2] ** 2)
                               / (2 * r.train_sigma_lin_vel ** 2))
        tracking_yaw = jnp.exp(-(ang_err[2] ** 2)
                               / (2 * r.train_sigma_ang_vel ** 2))
        upright_pen = jnp.sum(gravity_b[:2] ** 2)
        z_vel_pen = lin_vel_b[2] ** 2
        rp_ang_pen = jnp.sum(ang_vel_b[:2] ** 2)
        torque_pen = jnp.sum(torque ** 2)
        action_rate_pen = jnp.sum((torque - state.prev_torque) ** 2)

        tracking = r.w_tracking_lin * tracking_lin + r.w_tracking_yaw * tracking_yaw
        penalties = -(
            r.w_upright * upright_pen
            + r.w_z_vel * z_vel_pen
            + r.w_roll_pitch_ang * rp_ang_pen
            + r.w_torque * torque_pen
            + r.w_action_rate * action_rate_pen
        )
        reward = tracking + penalty_scale * penalties

        # --- termination ----------------------------------------------------
        base_h = data.qpos[2]
        fell = (base_h < self.ecfg.min_base_height) | \
               (gravity_b[2] > self.ecfg.upright_z_thresh)
        # explicit negative signal for falling (not curriculum-scaled)
        reward = reward - r.w_termination * fell.astype(jnp.float32)
        step = state.step + 1
        timeout = step >= self.ecfg.episode_length
        done = (fell | timeout).astype(jnp.float32)

        metrics = jnp.stack([tracking_lin, tracking_yaw, -penalties])

        e = self._privileged(model, motor_strength)

        hist_row = jnp.concatenate([obs, action])
        history = jnp.concatenate([state.history[1:], hist_row[None]], axis=0)

        return State(
            data=data, obs=obs, history=history, e=e, command=command,
            prev_action=action, prev_torque=torque,
            kp=kp, kd=kd, motor_strength=motor_strength,
            step=step, reward=reward, done=done, metrics=metrics, rng=rng,
        )


# --------------------------------------------------------------------------
# Domain randomization: build a vmapped batch of MJX models (model-level DR).
# --------------------------------------------------------------------------
def randomize_models(env: Go2Env, rng, num_envs: int):
    """Return (batched_model, in_axes) with per-env friction / mass / COM."""
    cfg = env.ecfg
    base = env.mjx_model
    foot_geoms = np.asarray(env.layout.foot_geom_ids)
    trunk = env.layout.trunk_body_id

    k_fric, k_mass, k_com = jax.random.split(rng, 3)
    friction = jax.random.uniform(k_fric, (num_envs,),
                                  minval=cfg.friction_range[0],
                                  maxval=cfg.friction_range[1])
    payload = jax.random.uniform(k_mass, (num_envs,),
                                 minval=cfg.payload_range[0],
                                 maxval=cfg.payload_range[1])
    com_xy = jax.random.uniform(k_com, (num_envs, 2),
                                minval=cfg.com_range[0],
                                maxval=cfg.com_range[1])

    gf = jnp.broadcast_to(base.geom_friction,
                          (num_envs,) + base.geom_friction.shape)
    gf = gf.at[:, foot_geoms, 0].set(friction[:, None])

    bm = jnp.broadcast_to(base.body_mass, (num_envs,) + base.body_mass.shape)
    bm = bm.at[:, trunk].add(payload)

    bipos = jnp.broadcast_to(base.body_ipos, (num_envs,) + base.body_ipos.shape)
    bipos = bipos.at[:, trunk, 0:2].add(com_xy)

    batched = base.tree_replace({
        "geom_friction": gf,
        "body_mass": bm,
        "body_ipos": bipos,
    })

    in_axes = jax.tree_util.tree_map(lambda _: None, base)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "body_mass": 0,
        "body_ipos": 0,
    })
    return batched, in_axes
