"""RMA deployment Controller for the gym-quadruped Project-3 benchmark.

Implements the required interface:

    class Controller:
        def reset(self, seed=None): ...
        def act(self, observation): return action, info

The controller wraps the two RMA networks trained in MJX:
  * base policy pi  (Phase 1) -- runs every control step,
  * adaptation module phi (Phase 2) -- refreshes the extrinsics estimate z_hat
    asynchronously (every ``async_every`` steps, ~10 Hz), exactly as in the RMA
    deployment loop.

It consumes ONLY observations the proprioceptive benchmark exposes:
    gravity_vector:base (3), imu_gyro (3), qpos_js (12), qvel_js (12),
    base_lin_vel_err:base (3), base_ang_vel_err:base (3).
These are assembled into the same 36-d state ``x_t`` used during training, in
the same joint order (qpos order: FL, FR, RL, RR).

The policy outputs a residual joint-position target ``a``; we convert it to a
joint torque with a fixed PD law (matching the MJX env), then permute the torque
into the actuator/ctrl order gym-quadruped applies (FR, FL, RR, RL).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import mujoco

from .config import Config
from .models import networks
from .envs.go2_constants import (
    NOMINAL_POSE, TORQUE_LIMIT, ctrl_from_qpos_permutation, resolve_model_path,
    resolve_layout,
)
from .utils import load_pytree


def _nominal_privileged(model_path):
    """Nominal e_t (17-dim) for the DEFAULT robot -- what no_adapt assumes:
    base trunk mass, base COM, motor strength 1.0, base foot friction, flat ground.
    Layout matches training (go2_env._privileged)."""
    m = mujoco.MjModel.from_xml_path(resolve_model_path(model_path))
    layout = resolve_layout(m)
    payload = float(m.body_mass[layout.trunk_body_id])
    com = np.asarray(m.body_ipos[layout.trunk_body_id, :2], dtype=np.float32)
    friction = float(m.geom_friction[layout.foot_geom_ids[0], 0])
    return np.concatenate([[payload], com, np.ones(12, np.float32),
                           [friction], [0.0]]).astype(np.float32)


def _ctrl_permutation(model_path):
    """Permutation mapping a qpos-ordered torque vector to the env's ctrl order.

    Derived from the *evaluation* model so it is correct regardless of how the
    Go2 MJCF orders its actuators (gym-quadruped's bundled model is identity;
    the Unitree mujoco model is FR,FL,RR,RL). tau_ctrl = tau_qpos[perm].
    """
    m = mujoco.MjModel.from_xml_path(resolve_model_path(model_path))
    return ctrl_from_qpos_permutation(m)

# Observation keys consumed from the benchmark dict (proprioceptive variant).
OBS_KEYS = (
    "gravity_vector:base",
    "imu_gyro",
    "qpos_js",
    "qvel_js",
    "base_lin_vel_err:base",
    "base_ang_vel_err:base",
)


class Controller:
    def __init__(
        self,
        phase1_ckpt: str = "checkpoints/phase1_final.pkl",
        phase2_ckpt: str | None = "checkpoints/phase2_final.pkl",
        cfg: Config | None = None,
        kp: float = 30.0,
        kd: float = 0.7,
        async_every: int = 10,
        control_decimation: int | None = None,
        mode: str = "rma",
    ):
        """Args:
        phase1_ckpt: base-policy + encoder params.
        phase2_ckpt: adaptation-module params (required for mode='rma').
        kp, kd: PD gains used to turn the residual position target into torque.
        async_every: refresh z_hat every this many *policy* updates (~10 Hz).
        control_decimation: env steps per policy update. gym-quadruped steps at
            500 Hz; the policy was trained at 50 Hz, so the default (from cfg)
            holds each residual target for 10 env steps while the PD torque is
            recomputed every step. Set to 1 if you call act() at the policy rate.
        mode: 'rma' (phi estimates z_hat from history), 'no_adapt' (z = mu(e_nominal),
            the lower bound), or 'expert' (z = mu(e_t) from the sim's true
            privileged factors, the upper bound -- call set_privileged(e_t) each
            episode to supply e_t).
        """
        self.cfg = cfg or Config()
        self.kp = float(kp)
        self.kd = float(kd)
        self.async_every = int(async_every)
        self.decimation = int(control_decimation if control_decimation is not None
                              else self.cfg.env.control_decimation)
        self.mode = mode

        self.nominal = np.asarray(NOMINAL_POSE, dtype=np.float32)
        self.tau_limit = np.asarray(TORQUE_LIMIT, dtype=np.float32)
        # ctrl-order permutation derived from the evaluation model (identity for
        # the gym-quadruped bundled Go2).
        self._perm = _ctrl_permutation(self.cfg.model_path)
        self.action_scale = self.cfg.env.action_scale
        self.k = self.cfg.env.history_len
        self.row_dim = self.cfg.net.state_dim + self.cfg.net.action_dim

        encoder, policy, _, adapt = networks.build_networks(self.cfg)
        p1 = load_pytree(phase1_ckpt)
        self._enc_params = p1["encoder"]
        self._pol_params = p1["policy"]

        # mu is always available: no_adapt uses mu(e_nominal); expert uses mu(e_t) with
        # the sim's true env factors, supplied per episode via set_privileged().
        @jax.jit
        def _mu(e):
            return encoder.apply(self._enc_params, e)

        self._mu = _mu

        if mode == "rma":
            if not phase2_ckpt:
                raise ValueError("mode='rma' requires a phase2 checkpoint")
            self._phi_params = load_pytree(phase2_ckpt)

            @jax.jit
            def _phi(history):
                return adapt.apply(self._phi_params, history)

            self._phi = _phi
        else:  # no_adapt (z = mu(e_nominal)) or expert (z = mu(e_t), set externally)
            # no_adapt assumes the DEFAULT robot and never changes z.
            e_nom = _nominal_privileged(self.cfg.model_path)
            self._z_fixed = np.asarray(self._mu(jnp.asarray(e_nom)))

        @jax.jit
        def _policy_mean(x, a_prev, z):
            mean, _ = policy.apply(self._pol_params, x, a_prev, z)
            return mean

        self._policy_mean = _policy_mean

        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None):
        self._history = None                 # lazily filled on first act
        self._prev_action = np.zeros(self.cfg.net.action_dim, dtype=np.float32)
        self._target_q = self.nominal.copy()  # PD setpoint (qpos order)
        self._z_async = None
        self._step = 0                        # env-step counter
        self._policy_updates = 0

    # ------------------------------------------------------------- expert mode
    def set_privileged(self, e_t):
        """Expert/oracle upper bound: set z = mu(e_t) from the sim's true env
        factors (17-dim, same layout as training). Call once per episode."""
        self._z_fixed = np.asarray(self._mu(jnp.asarray(e_t, dtype=jnp.float32)))

    # ------------------------------------------------------------------- act
    def _build_x(self, obs: dict) -> np.ndarray:
        for key in OBS_KEYS:
            if key not in obs:
                raise KeyError(
                    f"observation missing '{key}'. The benchmark must expose the "
                    f"proprioceptive obs set: {OBS_KEYS}")
        x = np.concatenate([
            np.asarray(obs["gravity_vector:base"], dtype=np.float32),
            np.asarray(obs["imu_gyro"], dtype=np.float32),
            np.asarray(obs["qpos_js"], dtype=np.float32),
            np.asarray(obs["qvel_js"], dtype=np.float32),
            np.asarray(obs["base_lin_vel_err:base"], dtype=np.float32),
            np.asarray(obs["base_ang_vel_err:base"], dtype=np.float32),
        ])
        assert x.shape[0] == self.cfg.net.state_dim, (
            f"assembled state dim {x.shape[0]} != {self.cfg.net.state_dim}")
        return x

    def act(self, observation):
        qpos_js = np.asarray(observation["qpos_js"], dtype=np.float32)
        qvel_js = np.asarray(observation["qvel_js"], dtype=np.float32)

        # --- policy + adaptation update (every `decimation` env steps) ------
        if self._step % self.decimation == 0:
            x = self._build_x(observation)

            # history row pairs the resulting obs with the action that caused it
            row = np.concatenate([x, self._prev_action])
            if self._history is None:
                self._history = np.tile(row, (self.k, 1)).astype(np.float32)
            else:
                self._history = np.roll(self._history, -1, axis=0)
                self._history[-1] = row

            if self.mode == "rma":
                if (self._z_async is None
                        or self._policy_updates % self.async_every == 0):
                    self._z_async = np.asarray(
                        self._phi(jnp.asarray(self._history)))
                z = self._z_async
            else:
                z = self._z_fixed

            a = np.asarray(self._policy_mean(
                jnp.asarray(x), jnp.asarray(self._prev_action), jnp.asarray(z)))
            self._target_q = self.nominal + self.action_scale * a
            self._prev_action = a
            self._policy_updates += 1

        # --- PD -> torque (recomputed every env step, qpos order) -----------
        tau = self.kp * (self._target_q - qpos_js) - self.kd * qvel_js
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        # --- permute to actuator / ctrl order -------------------------------
        action = tau[self._perm].astype(np.float32)

        self._step += 1
        info = {"z_hat": (self._z_async if self.mode == "rma" else self._z_fixed),
                "residual_action": self._prev_action}
        return action, info
