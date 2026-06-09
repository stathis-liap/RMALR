"""MJX evaluation rollout for RMA (sanity check inside the training simulator).

Reproduces the deployment loop (RMA Fig. 2B): the base policy runs at the
control rate; the adaptation module phi refreshes the extrinsics estimate z_hat
asynchronously at 1/10 of that rate. Runs a batch of domain-randomized MJX envs
and reports velocity-tracking metrics.

NOTE: this evaluates inside the *MJX* Go2 env. The official Project-3 grading
runs inside gym-quadruped -- use ``python -m rma.eval_gym`` for that.

Modes:
  rma       -> z_hat = phi(history), refreshed every 10 steps (default)
  expert    -> z = mu(e)  (privileged upper bound)
  no_adapt  -> z = mu(0)  (base policy, adaptation disabled)

Usage:
  python -m rma.evaluate --phase1 ckpt/phase1_final.pkl \
                         --phase2 ckpt/phase2_final.pkl --mode rma
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from .config import Config
from .envs.go2_env import Go2Env, randomize_models, OBS_SLICES
from .models import networks
from .utils import load_pytree


def evaluate(cfg, model_path, phase1_ckpt, phase2_ckpt, mode, num_envs, seed):
    env = Go2Env(cfg, model_path)
    encoder, policy, _, adapt = networks.build_networks(cfg)
    p1 = load_pytree(phase1_ckpt)
    phi = load_pytree(phase2_ckpt) if (mode == "rma" and phase2_ckpt) else None

    rng = jax.random.PRNGKey(seed)
    rng, k_dr, k_reset = jax.random.split(rng, 3)
    batched_model, in_axes = randomize_models(env, k_dr, num_envs)
    v_reset = jax.vmap(env.reset, in_axes=(in_axes, 0))
    v_step = jax.vmap(env.step, in_axes=(in_axes, 0, 0, None))

    async_every = 10  # phi at 1/10 of control rate

    def get_z(state):
        if mode == "expert":
            return encoder.apply(p1["encoder"], state.e)
        if mode == "no_adapt":
            return encoder.apply(p1["encoder"], jnp.zeros_like(state.e))
        return adapt.apply(phi, state.history)  # rma

    state = v_reset(batched_model, jax.random.split(k_reset, num_envs))
    z_async = get_z(state)

    def body(carry, t):
        state, z_async = carry
        do_refresh = (t % async_every) == 0
        z_new = get_z(state)
        z_async = jnp.where(do_refresh, z_new, z_async)

        mean, _ = policy.apply(p1["policy"], state.obs, state.prev_action, z_async)
        state = v_step(batched_model, state, mean, 1.0)

        lin_err = state.obs[:, OBS_SLICES["lin_vel_err"]]
        ang_err = state.obs[:, OBS_SLICES["ang_vel_err"]]
        lin_err_norm = jnp.linalg.norm(lin_err[:, :2], axis=-1)
        yaw_err_abs = jnp.abs(ang_err[:, 2])
        return (state, z_async), (state.reward, state.done, lin_err_norm, yaw_err_abs)

    (state, _), (rewards, dones, lin_err, yaw_err) = jax.lax.scan(
        body, (state, z_async), jnp.arange(cfg.env.episode_length))

    T = cfg.env.episode_length
    fell = dones > 0.5
    first_fall = jnp.where(fell.any(0), jnp.argmax(fell, axis=0), T)
    ttf = first_fall / T
    alive = jnp.cumsum(fell, axis=0) == 0          # alive up to first fall
    success = (ttf >= 0.99)

    denom = jnp.sum(alive)
    mean_reward = jnp.sum(jnp.where(alive, rewards, 0.0)) / denom
    mean_lin_err = jnp.sum(jnp.where(alive, lin_err, 0.0)) / denom
    mean_yaw_err = jnp.sum(jnp.where(alive, yaw_err, 0.0)) / denom

    print(f"=== mode={mode} envs={num_envs} ===")
    print(f"survival rate     : {float(success.mean()) * 100:.1f}%")
    print(f"TTF (norm)        : {float(ttf.mean()):.3f}")
    print(f"reward/step       : {float(mean_reward):.4f}")
    print(f"lin-vel track err : {float(mean_lin_err):.4f} m/s")
    print(f"yaw-rate track err: {float(mean_yaw_err):.4f} rad/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", type=str, default="checkpoints/phase1_final.pkl")
    ap.add_argument("--phase2", type=str, default="checkpoints/phase2_final.pkl")
    ap.add_argument("--mode", type=str, default="rma",
                    choices=["rma", "expert", "no_adapt"])
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.model is not None:
        cfg.model_path = args.model
    evaluate(cfg, cfg.model_path, args.phase1, args.phase2,
             args.mode, args.envs, args.seed)


if __name__ == "__main__":
    main()
