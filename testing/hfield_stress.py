"""Run stress scenarios on the MJX training heightfield (vmapped batch).

Mirrors ``testing.runner`` but on the real training terrain: each scenario's
stressors (friction, payload, COM, motor strength, pushes) are pinned into the
MJX env's domain randomization, a batch of envs is rolled out in parallel, and
survival + tracking are reported. This is the faithful "stress on terrain" path
-- the policy trained on this exact heightfield, unlike gym-quadruped's over-tall
perlin where the legs clip through.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rma.config import Config
from rma.envs.go2_env import Go2Env, randomize_models, OBS_SLICES
from rma.models import networks
from rma.utils import load_pytree


def _cfg_for(sc, z_scale):
    """Config with the scenario's stressors pinned into the MJX env's DR."""
    cfg = Config()
    cfg.env.terrain = "hfield"
    cfg.env.fractal_z_scale = z_scale
    cfg.env.friction_range = tuple(sc.friction)
    cfg.env.payload_range = (sc.payload, sc.payload)
    cfg.env.com_range = (sc.com_shift[0], sc.com_shift[0])   # x-shift (also y; minor)
    cfg.env.motor_strength_range = (sc.motor_scale, sc.motor_scale)
    cfg.env.kp_range = (30.0, 30.0)                          # nominal fixed PD
    cfg.env.kd_range = (0.7, 0.7)
    cfg.env.push_prob = (1.0 / sc.push_interval) if sc.push_interval else 0.0
    cfg.env.push_lin_vel = sc.push_vel
    cfg.env.resample_prob = 0.0          # keep the stressor fixed within an episode
    cfg.env.cmd_resample_prob = 0.01     # but vary the velocity command
    return cfg


def run_scenario_hfield(sc, phase1, phase2, mode, n_envs, max_steps, seed, z_scale=0.10):
    """Roll out `n_envs` MJX envs under the scenario; return a summary row."""
    cfg = _cfg_for(sc, z_scale)
    cfg.env.episode_length = max_steps + 1                   # only falls end it
    env = Go2Env(cfg, "auto")
    enc, pol, _, adapt = networks.build_networks(cfg)
    p1 = load_pytree(phase1)
    phi = load_pytree(phase2) if mode == "rma" else None

    rng = jax.random.PRNGKey(seed)
    rng, k_dr, k_reset = jax.random.split(rng, 3)
    batched, in_axes = randomize_models(env, k_dr, n_envs)
    v_reset = jax.vmap(env.reset, in_axes=(in_axes, 0))
    v_step = jax.vmap(env.step, in_axes=(in_axes, 0, 0, None))

    def get_z(state):
        if mode == "expert":
            return enc.apply(p1["encoder"], state.e)
        if mode == "no_adapt":
            return enc.apply(p1["encoder"], jnp.zeros_like(state.e))
        return adapt.apply(phi, state.history)

    @jax.jit
    def rollout(state, z0):
        def body(carry, t):
            state, z = carry
            z = jnp.where((t % 10) == 0, get_z(state), z)    # phi at ~10 Hz
            mean, _ = pol.apply(p1["policy"], state.obs, state.prev_action, z)
            state = v_step(batched, state, mean, 1.0)
            lin = jnp.linalg.norm(state.obs[:, OBS_SLICES["lin_vel_err"]][:, :2], axis=-1)
            yaw = jnp.abs(state.obs[:, OBS_SLICES["ang_vel_err"]][:, 2])
            return (state, z), (state.done, lin, yaw)
        _, out = jax.lax.scan(body, (state, z0), jnp.arange(max_steps))
        return out

    state = v_reset(batched, jax.random.split(k_reset, n_envs))
    dones, lin, yaw = (np.asarray(x) for x in rollout(state, get_z(state)))

    fell = dones > 0.5                                       # (T, n_envs)
    first_fall = np.where(fell.any(0), np.argmax(fell, 0), max_steps)
    alive = np.cumsum(fell, 0) == 0                          # alive up to first fall
    denom = np.maximum(alive.sum(0), 1)
    # a fallen env (esp. extreme payload) can diverge to NaN after it falls; the
    # alive mask drops those steps, but zero them first so nan*0 doesn't leak.
    lin = np.nan_to_num(lin, nan=0.0)
    yaw = np.nan_to_num(yaw, nan=0.0)
    return {
        "scenario": sc.name,
        "mode": mode,
        "survival": 100.0 * np.mean(first_fall >= 0.99 * max_steps),
        "steps": float(np.mean(first_fall)),
        "lin_err": float(np.mean((lin * alive).sum(0) / denom)),
        "yaw_err": float(np.mean((yaw * alive).sum(0) / denom)),
    }
