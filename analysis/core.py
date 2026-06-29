"""Shared engine for the RMA analysis figures.

Everything downstream (tracking, robustness, adaptation figures) is built on one
batched MJX rollout primitive that runs the trained networks in the exact three
RMA evaluation modes:

    no_adapt  z = mu(e_nom)   base policy assumes the default robot   -> LOWER bound
    rma       z = phi(history) adaptation module from history  -> the method
    expert    z = mu(e_t)      sim's true privileged factors   -> UPPER bound

(matching ``rma.evaluate`` / the paper's Robust / RMA / Expert baselines).

Why MJX and not gym-quadruped here: the analysis needs batched access to the
*latent extrinsics* z, the *true* e_t, the per-foot contacts and the per-env
ground-truth factors -- none of which the grader exposes. The MJX ``Go2Env``
mirrors the Project-3 observation, reward and PD path used by the deployed
``Controller`` (and the per-env friction/payload/COM in ``randomize_models`` let
us sweep a whole axis in a single rollout). The faithful grader cross-check is
``python -m rma.eval_gym``.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from rma.config import Config
from rma.envs.go2_env import Go2Env, OBS_SLICES
from rma.models import networks
from rma.utils import load_pytree

# ---------------------------------------------------------------------------
# Presentation: one colour / label per mode, used by every figure.
# ---------------------------------------------------------------------------
MODES = ("no_adapt", "rma", "expert")
MODE_COLOR = {"no_adapt": "#d1495b", "rma": "#2e86ab", "expert": "#3c8c40"}
MODE_LABEL = {
    "no_adapt": "no-adapt  (z=μ(eₙₒₘ), lower bound)",
    "rma":      "RMA  (z=φ(history))",
    "expert":   "expert  (z=μ(eₜ), upper bound)",
}
MODE_LABEL_SHORT = {"no_adapt": "no-adapt", "rma": "RMA", "expert": "expert"}

CKPT1 = "checkpoints/phase1_final.pkl"
CKPT2 = "checkpoints/phase2_final.pkl"
CACHE_DIR = "figures/cache"
FIG_DIR = "figures"


# ---------------------------------------------------------------------------
# Per-env factor grids (collapse a sweep axis into one batched rollout).
# ---------------------------------------------------------------------------
def _build_batched(env, friction, payload, com_xy):
    """Batched MJX model with explicit per-env friction / payload / COM arrays.

    Same leaves as ``randomize_models`` but deterministic, so a sweep axis can be
    laid out across the env batch and binned afterwards.
    """
    base = env.mjx_model
    foot = np.asarray(env.layout.foot_geom_ids)
    trunk = env.layout.trunk_body_id
    n = friction.shape[0]

    gf = jnp.broadcast_to(base.geom_friction, (n,) + base.geom_friction.shape)
    gf = gf.at[:, foot, 0].set(jnp.asarray(friction)[:, None])
    bm = jnp.broadcast_to(base.body_mass, (n,) + base.body_mass.shape)
    bm = bm.at[:, trunk].add(jnp.asarray(payload))
    bip = jnp.broadcast_to(base.body_ipos, (n,) + base.body_ipos.shape)
    bip = bip.at[:, trunk, 0:2].add(jnp.asarray(com_xy))

    batched = base.tree_replace(
        {"geom_friction": gf, "body_mass": bm, "body_ipos": bip})
    in_axes = jax.tree_util.tree_map(lambda _: None, base)
    in_axes = in_axes.tree_replace(
        {"geom_friction": 0, "body_mass": 0, "body_ipos": 0})
    return batched, in_axes


# ---------------------------------------------------------------------------
# Result container.
# ---------------------------------------------------------------------------
@dataclass
class Rollout:
    command: np.ndarray     # (T, N, 3)  commanded [vx, vy, yaw]
    achieved: np.ndarray    # (T, N, 3)  achieved base-frame [vx, vy, yaw]
    lin_err: np.ndarray     # (T, N)     ||xy velocity error||
    yaw_err: np.ndarray     # (T, N)     |yaw-rate error|
    z: np.ndarray           # (T, N, 8)  latent extrinsics actually used
    e: np.ndarray           # (N, 17)    true privileged factors (at reset)
    torque: np.ndarray      # (T, N, 12) applied joint torque
    reward: np.ndarray      # (T, N)     per-step Project-3 reward
    done: np.ndarray        # (T, N)
    alive: np.ndarray       # (T, N) bool: alive up to (and incl.) first fall
    factors: dict           # per-env ground truth: friction/payload/com_x/motor/...

    @property
    def T(self):
        return self.command.shape[0]

    @property
    def N(self):
        return self.command.shape[1]

    def survival(self):
        """Fraction of envs that never fell (per-env bool, shape (N,))."""
        return ~np.any(self.done > 0.5, axis=0)

    def ep_lin_err(self):
        """Per-env mean xy tracking error over the alive portion (N,)."""
        a = self.alive
        return (self.lin_err * a).sum(0) / np.maximum(a.sum(0), 1)

    def ep_yaw_err(self):
        a = self.alive
        return (self.yaw_err * a).sum(0) / np.maximum(a.sum(0), 1)

    def ttf_norm(self):
        """Normalized time-to-fall in [0,1] (1 = never fell), per env (N,)."""
        fell = self.done > 0.5
        first = np.where(fell.any(0), np.argmax(fell, 0), self.T)
        return first / self.T

    def smoothness(self):
        """Mean ||tau_t - tau_{t-1}||^2 over alive steps, per env (N,)."""
        dtau = np.diff(self.torque, axis=0)
        s = (dtau ** 2).sum(-1)                       # (T-1, N)
        a = self.alive[1:]
        return (s * a).sum(0) / np.maximum(a.sum(0), 1)

    def torque_sq(self):
        """Mean ||tau||^2 over alive steps, per env (N,)."""
        s = (self.torque ** 2).sum(-1)
        a = self.alive
        return (s * a).sum(0) / np.maximum(a.sum(0), 1)

    def reward_per_step(self):
        a = self.alive
        return (np.nan_to_num(self.reward) * a).sum(0) / np.maximum(a.sum(0), 1)


class Engine:
    """Loads the trained networks once; rolls out batches in any mode."""

    def __init__(self, ckpt1=CKPT1, ckpt2=CKPT2, terrain="scene", z_scale=0.10):
        self.ckpt1, self.ckpt2 = ckpt1, ckpt2
        self.cfg = Config()
        self.cfg.env.terrain = terrain          # "scene"=flat grader terrain
        self.cfg.env.fractal_z_scale = z_scale
        # Deployment-faithful, deterministic episode: fixed PD (the Controller
        # uses kp=30/kd=0.7), no within-episode DR churn / pushes / cmd resample
        # unless a figure asks. Long horizon so only a fall ends an episode.
        self.cfg.env.kp_range = (30.0, 30.0)
        self.cfg.env.kd_range = (0.7, 0.7)
        self.cfg.env.resample_prob = 0.0
        self.cfg.env.push_prob = 0.0
        self.cfg.env.cmd_resample_prob = 0.0
        self.cfg.env.episode_length = 10 ** 9
        self.env = Go2Env(self.cfg, "auto")
        enc, pol, _, adapt = networks.build_networks(self.cfg)
        p1 = load_pytree(ckpt1)
        self._p1 = p1
        self._enc, self._pol, self._adapt = enc, pol, adapt
        self._phi = load_pytree(ckpt2) if ckpt2 and os.path.exists(ckpt2) else None
        self.edim = self.cfg.net.env_factor_dim
        self._jit_cache = {}
        # no_adapt extrinsics: the policy "thinks the robot is nominal and never
        # changes" -- so z is the constant mu(e_nominal) for the DEFAULT robot
        # (base trunk mass, no COM shift, motor strength 1.0, base foot friction,
        # flat ground), not mu(0) (which feeds the encoder an out-of-distribution
        # zero friction / zero motor strength it never saw in training).
        self.e_nominal = jnp.asarray(self.env._privileged(
            self.env.mjx_model, jnp.ones(self.cfg.net.action_dim), jnp.zeros(())))
        self.z_nominal = self._enc.apply(self._p1["encoder"], self.e_nominal)  # (8,)

    # -- the three extrinsics estimators -----------------------------------
    def _z_of(self, mode, state):
        if mode == "expert":
            return self._enc.apply(self._p1["encoder"], state.e)
        if mode == "no_adapt":                                    # constant mu(e_nominal)
            return jnp.broadcast_to(self.z_nominal, state.e.shape[:-1] + (self.z_nominal.shape[-1],))
        return self._adapt.apply(self._phi, state.history)        # rma

    def mu(self, e):
        """mu(e): encode a privileged factor vector (or batch) to z."""
        return np.asarray(self._enc.apply(self._p1["encoder"], jnp.asarray(e)))

    def phi(self, history):
        """phi(history): adaptation-module estimate of z from a state/action window."""
        return np.asarray(self._adapt.apply(self._phi, jnp.asarray(history)))

    # -- main primitive -----------------------------------------------------
    def rollout(self, mode, T, friction, payload, com_xy, motor, command,
                seed=0, push_mag=None, push_period=0, obs_noise=0.0):
        """Roll out one batch. All per-env factor args are arrays of length N.

        command     : (T, N, 3) scripted command fed every step (constant-in-t for
                      "hold a command" experiments).
        push_mag    : (N,) per-env shove speed (m/s); push_period: steps between
                      shoves (0 = no pushes). A random horizontal base-velocity
                      impulse, like the env's training pushes / the hidden tests.
        obs_noise   : std of zero-mean Gaussian noise added to the proprioceptive
                      state x_t fed to the base policy (sensor-noise hidden test).
        Returns a ``Rollout``.
        """
        if mode == "rma" and self._phi is None:
            raise RuntimeError("rma mode needs a phase-2 checkpoint")
        env = self.env
        N = friction.shape[0]
        friction = np.asarray(friction, np.float32)
        payload = np.asarray(payload, np.float32)
        com_xy = np.asarray(com_xy, np.float32).reshape(N, 2)
        motor = np.asarray(motor, np.float32)
        if motor.ndim == 1:
            motor = np.repeat(motor[:, None], 12, axis=1)
        command = jnp.asarray(command, jnp.float32)
        push_mag = (jnp.zeros(N) if push_mag is None
                    else jnp.asarray(push_mag, jnp.float32))
        push_period = jnp.int32(push_period)
        obs_noise = jnp.float32(obs_noise)

        batched, in_axes = _build_batched(env, friction, payload, com_xy)
        v_reset = jax.vmap(env.reset, in_axes=(in_axes, 0))
        v_step = jax.vmap(env.step, in_axes=(in_axes, 0, 0, None))

        enc, pol, adapt = self._enc, self._pol, self._adapt
        p1, phi = self._p1, self._phi
        z_nominal = self.z_nominal

        def get_z(state):
            if mode == "expert":
                return enc.apply(p1["encoder"], state.e)
            if mode == "no_adapt":                # constant mu(e_nominal), all envs
                return jnp.broadcast_to(z_nominal, state.e.shape[:-1] + z_nominal.shape)
            return adapt.apply(phi, state.history)

        def run(batched, state, cmd, pmag, pperiod, onoise, key):
            def body(carry, inp):
                state, z = carry
                cmd_t, t = inp
                # optional external shove (random horizontal direction)
                do_push = (pperiod > 0) & ((t % jnp.maximum(pperiod, 1)) == 0)
                d = jax.random.normal(jax.random.fold_in(key, t), (state.obs.shape[0], 3))
                d = d.at[:, 2].set(0.0)
                d = d / (jnp.linalg.norm(d, axis=-1, keepdims=True) + 1e-6)
                dv = jnp.where(do_push, pmag[:, None] * d, 0.0)
                qvel = state.data.qvel.at[:, 0:3].add(dv)
                state = state.replace(data=state.data.replace(qvel=qvel))

                z = jnp.where((t % 10) == 0, get_z(state), z)      # phi at ~10 Hz
                # optional proprioceptive sensor noise on the policy's x_t input
                nk = jax.random.fold_in(key, t + 100003)
                obs_in = state.obs + onoise * jax.random.normal(nk, state.obs.shape)
                mean, _ = pol.apply(p1["policy"], obs_in, state.prev_action, z)
                state = state.replace(command=cmd_t)
                state = v_step(batched, state, mean, 1.0)
                rec = (state.command,
                       state.obs[:, OBS_SLICES["lin_vel_err"]],
                       state.obs[:, OBS_SLICES["ang_vel_err"]],
                       z, state.prev_torque, state.reward, state.done)
                return (state, z), rec
            z0 = get_z(state)
            _, out = jax.lax.scan(body, (state, z0), (cmd, jnp.arange(cmd.shape[0])))
            return out

        run = self._jit_cache.setdefault(mode, jax.jit(run))

        state = v_reset(batched, jax.random.split(jax.random.PRNGKey(seed), N))
        state = state.replace(motor_strength=jnp.asarray(motor))
        key = jax.random.PRNGKey(seed + 777)
        cmd_rec, lin_err, ang_err, z, torque, reward, done = (
            np.asarray(x) for x in jax.block_until_ready(
                run(batched, state, command, push_mag, push_period, obs_noise, key)))

        # achieved = command - error (componentwise, base frame)
        achieved = np.stack([
            cmd_rec[..., 0] - lin_err[..., 0],
            cmd_rec[..., 1] - lin_err[..., 1],
            cmd_rec[..., 2] - ang_err[..., 2],
        ], axis=-1)
        lin = np.nan_to_num(np.linalg.norm(lin_err[..., :2], axis=-1), nan=0.0)
        yaw = np.nan_to_num(np.abs(ang_err[..., 2]), nan=0.0)
        # a fallen env can diverge to NaN *after* it fell; the alive mask drops
        # those steps, but zero them first so nan*0 doesn't leak into the means.
        torque = np.nan_to_num(torque)
        reward = np.nan_to_num(reward)
        fell = done > 0.5
        alive = np.cumsum(fell, axis=0) == 0
        return Rollout(
            command=cmd_rec, achieved=achieved, lin_err=lin, yaw_err=yaw,
            z=z, e=np.asarray(state.e), torque=torque, reward=reward,
            done=done, alive=alive,
            factors=dict(friction=friction, payload=payload,
                         com_x=com_xy[:, 0], motor=motor[:, 0],
                         terrain_z=np.full(N, self.cfg.env.fractal_z_scale)),
        )


# ---------------------------------------------------------------------------
# Command schedules.
# ---------------------------------------------------------------------------
def const_commands(commands, T):
    """(N,3) per-env constant commands -> (T,N,3) schedule held over the episode."""
    commands = np.asarray(commands, np.float32)
    return np.broadcast_to(commands[None], (T,) + commands.shape).copy()


def sample_commands(n, seed=0, speed=(0.2, 0.8), yaw=(-0.8, 0.8)):
    """N representative in-distribution velocity commands (random heading)."""
    rng = np.random.default_rng(seed)
    sp = rng.uniform(*speed, n)
    th = rng.uniform(-np.pi, np.pi, n)
    wz = rng.uniform(*yaw, n)
    return np.stack([sp * np.cos(th), sp * np.sin(th), wz], axis=1).astype(np.float32)


def nominal_factors(n):
    """Per-env factor arrays at the nominal (in-distribution centre) setting."""
    return dict(friction=np.full(n, 1.0, np.float32),
                payload=np.zeros(n, np.float32),
                com_xy=np.zeros((n, 2), np.float32),
                motor=np.ones(n, np.float32))


# ---------------------------------------------------------------------------
# Small helpers shared by the figure modules.
# ---------------------------------------------------------------------------
def cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name + ".npz")


def fig_path(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)


def pca_2d(x, basis=None):
    """Project rows of x to 2-D. If basis given, reuse it (so two clouds share axes)."""
    x = np.asarray(x, np.float64)
    mean = x.mean(0)
    if basis is None:
        _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
        basis = (mean, vt[:2])
    m, comps = basis
    return (x - m) @ comps.T, basis


def timer(msg):
    class _T:
        def __enter__(self):
            self.t = time.time(); print(f"  {msg} ...", flush=True); return self
        def __exit__(self, *a):
            print(f"  {msg}: {time.time() - self.t:.1f}s", flush=True)
    return _T()
