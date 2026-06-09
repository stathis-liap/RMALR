"""Phase-2: train the adaptation module phi via on-policy supervised learning.

phi(history) -> z_hat regresses the true extrinsics z = mu(e). Training is
DAgger-style (Ross et al. 2011, as in the paper): roll out the frozen base
policy *driven by phi's own prediction* z_hat, and regress against the ground
truth z from the (frozen) encoder. The frozen base policy + encoder come from
Phase 1.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import optax

from ..envs.go2_env import Go2Env, randomize_models
from ..models import networks
from ..utils import save_pytree, load_pytree, tree_select


class AdaptTrainer:
    def __init__(self, cfg, model_path, phase1_ckpt):
        self.cfg = cfg
        self.acfg = cfg.adapt
        self.env = Go2Env(cfg, model_path)
        self.encoder, self.policy, _, self.adapt = networks.build_networks(cfg)

        self.p1 = load_pytree(phase1_ckpt)  # {'encoder','policy','value'}

        rng = jax.random.PRNGKey(self.acfg.seed)
        rng, k_dr = jax.random.split(rng)
        self.batched_model, self.in_axes = randomize_models(
            self.env, k_dr, self.acfg.num_envs)
        self._rng = rng

        self.v_reset = jax.vmap(self.env.reset, in_axes=(self.in_axes, 0))
        self.v_step = jax.vmap(self.env.step, in_axes=(self.in_axes, 0, 0, None))

    def init_phi(self, rng):
        nc = self.cfg.net
        hist = jnp.zeros((1, self.cfg.env.history_len, nc.state_dim + nc.action_dim))
        return self.adapt.init(rng, hist)

    def _action(self, phi_params, state):
        """Base-policy mean action driven by phi's predicted extrinsics."""
        z_hat = self.adapt.apply(phi_params, state.history)
        mean, _ = self.policy.apply(self.p1["policy"], state.obs,
                                    state.prev_action, z_hat)
        return mean, z_hat

    def rollout(self, phi_params, model, state, rng):
        def body(carry, _):
            state, rng = carry
            action, _ = self._action(phi_params, state)
            z_target = self.encoder.apply(self.p1["encoder"], state.e)
            hist_in = state.history  # input window for this step
            next_state = self.v_step(model, state, action, 1.0)
            reset_state = self.v_reset(model, next_state.rng)
            next_state = tree_select(next_state.done > 0.5, reset_state, next_state)
            return (next_state, rng), (hist_in, z_target)

        (state, rng), (hist, ztgt) = jax.lax.scan(
            body, (state, rng), None, length=self.acfg.unroll_length)
        return state, rng, hist, ztgt

    def update(self, phi_params, opt_state, hist, ztgt):
        a = self.acfg
        N = hist.shape[0]
        mb = N // a.num_minibatches
        perm = jnp.arange(N)

        def loss_fn(phi_params, h, z):
            z_hat = self.adapt.apply(phi_params, h)
            return jnp.mean(jnp.sum((z_hat - z) ** 2, axis=-1))

        def mb_step(carry, idx):
            phi_params, opt_state = carry
            sl = jax.lax.dynamic_slice_in_dim(perm, idx * mb, mb)
            loss, grads = jax.value_and_grad(loss_fn)(phi_params, hist[sl], ztgt[sl])
            updates, opt_state = self.opt.update(grads, opt_state, phi_params)
            phi_params = optax.apply_updates(phi_params, updates)
            return (phi_params, opt_state), loss

        (phi_params, opt_state), losses = jax.lax.scan(
            mb_step, (phi_params, opt_state), jnp.arange(a.num_minibatches))
        return phi_params, opt_state, losses.mean()

    def train(self):
        a = self.acfg
        rng = self._rng
        rng, k_init, k_reset = jax.random.split(rng, 3)
        phi_params = self.init_phi(k_init)

        self.opt = optax.adam(a.learning_rate)
        opt_state = self.opt.init(phi_params)

        rollout = jax.jit(self.rollout)
        update = jax.jit(self.update)

        env_keys = jax.random.split(k_reset, a.num_envs)
        state = self.v_reset(self.batched_model, env_keys)

        for it in range(a.num_iterations):
            t0 = time.time()
            rng, k_roll = jax.random.split(rng)
            state, _, hist, ztgt = rollout(
                phi_params, self.batched_model, state, k_roll)

            # flatten (T, N, ...) -> (T*N, ...)
            hist_f = hist.reshape((-1,) + hist.shape[2:])
            ztgt_f = ztgt.reshape((-1,) + ztgt.shape[2:])

            phi_params, opt_state, loss = update(
                phi_params, opt_state, hist_f, ztgt_f)

            if it % 10 == 0:
                sps = (a.num_envs * a.unroll_length) / (time.time() - t0)
                print(f"[adapt] it={it} mse={float(loss):.5f} steps/s={sps:.0f}")
            if it % a.save_every == 0 and it > 0:
                save_pytree(f"{self.cfg.checkpoint_dir}/phase2_{it}.pkl", phi_params)

        save_pytree(f"{self.cfg.checkpoint_dir}/phase2_final.pkl", phi_params)
        return phi_params
