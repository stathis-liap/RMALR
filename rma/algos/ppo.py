"""Phase-1 PPO: jointly train base policy pi and env-factor encoder mu.

Asymmetric actor-critic -- the policy sees z_t = mu(e_t), the value function sees
the raw privileged e_t -- on a vmapped batch of domain-randomized MJX envs. PPO
settings follow the paper/supplementary (clip 0.2, GAE lambda 0.95, 4 epochs x
4 minibatches, lr 5e-4, value-coef 0.5), plus a small entropy bonus that keeps
the policy exploring (see config.PPOConfig.entropy_coef).
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from ..envs.go2_env import Go2Env, randomize_models
from ..models import networks
from ..utils import save_pytree, tree_select


@struct.dataclass
class Transition:
    x: jnp.ndarray
    a_prev: jnp.ndarray
    e: jnp.ndarray
    action: jnp.ndarray
    logp: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    metrics: jnp.ndarray   # (3,) [tracking_lin, tracking_yaw, penalty_mag]


def make_tb_writer(logdir):
    """Optional tensorboardX writer (no-op if the package is missing)."""
    try:
        from tensorboardX import SummaryWriter
        return SummaryWriter(logdir)
    except ImportError:
        print(f"[tb] tensorboardX not installed; skipping TensorBoard logs "
              f"(would write to {logdir})")
        return None


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    def step(carry, inp):
        next_value, next_adv = carry
        reward, value, done = inp
        nonterm = 1.0 - done
        delta = reward + gamma * nonterm * next_value - value
        adv = delta + gamma * lam * nonterm * next_adv
        return (value, adv), adv

    (_, _), advs = jax.lax.scan(
        step, (last_value, jnp.zeros_like(last_value)),
        (rewards, values, dones), reverse=True,
    )
    return advs, advs + values


class PPOTrainer:
    def __init__(self, cfg, model_path):
        self.cfg = cfg
        self.pcfg = cfg.ppo
        self.env = Go2Env(cfg, model_path)
        self.encoder, self.policy, self.value, _ = networks.build_networks(cfg)

        # Build a fixed batch of domain-randomized models (model-level DR).
        rng = jax.random.PRNGKey(self.pcfg.seed)
        rng, k_dr = jax.random.split(rng)
        self.batched_model, self.in_axes = randomize_models(
            self.env, k_dr, self.pcfg.num_envs)
        self._rng = rng

        self.v_reset = jax.vmap(self.env.reset, in_axes=(self.in_axes, 0))
        self.v_step = jax.vmap(self.env.step, in_axes=(self.in_axes, 0, 0, None))

    # ----------------------------------------------------------- init params
    def init_params(self, rng):
        nc = self.cfg.net
        k1, k2, k3 = jax.random.split(rng, 3)
        x = jnp.zeros((1, nc.state_dim))
        a = jnp.zeros((1, nc.action_dim))
        e = jnp.zeros((1, nc.env_factor_dim))
        z = jnp.zeros((1, nc.latent_dim))
        params = {
            "encoder": self.encoder.init(k1, e),
            "policy": self.policy.init(k2, x, a, z),
            "value": self.value.init(k3, x, a, e),
        }
        return params

    # --------------------------------------------------------------- actor
    def actor_value(self, params, x, a_prev, e, key=None):
        z = self.encoder.apply(params["encoder"], e)
        mean, log_std = self.policy.apply(params["policy"], x, a_prev, z)
        value = self.value.apply(params["value"], x, a_prev, e)
        if key is None:
            return mean, value
        std = jnp.exp(log_std)
        action = mean + std * jax.random.normal(key, mean.shape)
        logp = networks.gaussian_log_prob(mean, log_std, action)
        return action, logp, value, mean, log_std

    # -------------------------------------------------------------- rollout
    def rollout(self, params, model, state, rng, penalty_scale):
        def body(carry, _):
            state, rng = carry
            rng, key = jax.random.split(rng)
            x, a_prev, e = state.obs, state.prev_action, state.e
            action, logp, value, _, _ = self.actor_value(params, x, a_prev, e, key)
            next_state = self.v_step(model, state, action, penalty_scale)
            done = next_state.done
            # capture reward/metrics BEFORE auto-reset, or terminal rewards
            # (incl. the fall penalty) would be replaced by the reset's zeros
            reward, metrics = next_state.reward, next_state.metrics
            # auto-reset done envs (reset uses each env's advanced rng)
            reset_state = self.v_reset(model, next_state.rng)
            next_state = tree_select(done > 0.5, reset_state, next_state)
            tr = Transition(x=x, a_prev=a_prev, e=e, action=action, logp=logp,
                            value=value, reward=reward, done=done,
                            metrics=metrics)
            return (next_state, rng), tr

        (state, rng), traj = jax.lax.scan(
            body, (state, rng), None, length=self.pcfg.unroll_length)
        _, last_value = self.actor_value(params, state.obs, state.prev_action, state.e)
        return state, rng, traj, last_value

    # --------------------------------------------------------------- update
    def update(self, params, opt_state, batch, advantages, returns):
        p = self.pcfg
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        N = batch.x.shape[0]
        mb = N // p.num_minibatches

        def loss_fn(params, b, adv_b, ret_b):
            # recompute log-prob / value under current params (no sampling)
            z = self.encoder.apply(params["encoder"], b.e)
            mean, log_std = self.policy.apply(params["policy"], b.x, b.a_prev, z)
            logp = networks.gaussian_log_prob(mean, log_std, b.action)
            value = self.value.apply(params["value"], b.x, b.a_prev, b.e)

            ratio = jnp.exp(logp - b.logp)
            unclipped = ratio * adv_b
            clipped = jnp.clip(ratio, 1 - p.clip_ratio, 1 + p.clip_ratio) * adv_b
            policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

            v_clipped = b.value + jnp.clip(value - b.value, -p.clip_ratio, p.clip_ratio)
            v_loss = jnp.maximum((value - ret_b) ** 2, (v_clipped - ret_b) ** 2)
            value_loss = 0.5 * jnp.mean(v_loss)

            # Entropy bonus keeps the policy exploring (std is state-independent,
            # so this is the same for every row but still pulls log_std up).
            entropy = jnp.mean(networks.gaussian_entropy(log_std))
            total = (policy_loss + p.value_loss_coef * value_loss
                     - p.entropy_coef * entropy)
            return total, (policy_loss, value_loss, entropy)

        def epoch(carry, key):
            params, opt_state = carry
            perm = jax.random.permutation(key, N)
            def mb_step(carry, idx):
                params, opt_state = carry
                sl = jax.lax.dynamic_slice_in_dim(perm, idx * mb, mb)
                b = jax.tree_util.tree_map(lambda x: x[sl], batch)
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    params, b, adv[sl], returns[sl])
                updates, opt_state = self.opt.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), (loss, aux[2])
            (params, opt_state), (mb_loss, mb_ent) = jax.lax.scan(
                mb_step, (params, opt_state), jnp.arange(p.num_minibatches))
            return (params, opt_state), (mb_loss.mean(), mb_ent.mean())

        keys = jax.random.split(jax.random.PRNGKey(0), p.num_epochs)
        (params, opt_state), (ep_loss, ep_ent) = jax.lax.scan(
            epoch, (params, opt_state), keys)
        return params, opt_state, ep_loss.mean(), ep_ent.mean()

    # ---------------------------------------------------------------- train
    def train(self):
        p = self.pcfg
        rng = self._rng
        rng, k_init, k_reset = jax.random.split(rng, 3)
        params = self.init_params(k_init)

        self.opt = optax.chain(
            optax.clip_by_global_norm(p.max_grad_norm),
            optax.adam(p.learning_rate, b1=0.9, b2=0.999, eps=1e-8),
        )
        opt_state = self.opt.init(params)

        # jit the bound methods (self captured -> no static self argument)
        rollout = jax.jit(self.rollout)
        update = jax.jit(self.update)

        env_keys = jax.random.split(k_reset, p.num_envs)
        state = self.v_reset(self.batched_model, env_keys)

        penalty_scale = self.cfg.reward.penalty_curriculum_k0
        decay = self.cfg.reward.penalty_curriculum_decay
        tb = make_tb_writer(f"{self.cfg.checkpoint_dir}/tb/phase1")

        for it in range(p.num_iterations):
            t0 = time.time()
            rng, k_roll = jax.random.split(rng)
            state, _, traj, last_value = rollout(
                params, self.batched_model, state, k_roll, penalty_scale)

            advantages, returns = compute_gae(
                traj.reward, traj.value, traj.done, last_value, p.gamma, p.gae_lambda)

            # flatten (T, N, ...) -> (T*N, ...)
            flat = jax.tree_util.tree_map(
                lambda x: x.reshape((-1,) + x.shape[2:]), traj)
            adv_f = advantages.reshape(-1)
            ret_f = returns.reshape(-1)

            params, opt_state, loss, entropy = update(
                params, opt_state, flat, adv_f, ret_f)

            penalty_scale = min(1.0, penalty_scale ** decay)

            if it % 10 == 0:
                mean_r = float(jnp.mean(traj.reward))
                sps = (p.num_envs * p.unroll_length) / (time.time() - t0)
                # diagnostics: reward decomposition + episode statistics
                m = np.asarray(jax.device_get(traj.metrics)).mean(axis=(0, 1))
                track_lin, track_yaw, pen_mag = float(m[0]), float(m[1]), float(m[2])
                done_rate = float(jnp.mean(traj.done))
                ep_len = 1.0 / max(done_rate, 1e-6)  # control steps, estimate
                # mean action std -> direct read on whether exploration is alive
                std = float(jnp.exp(jnp.mean(
                    params["policy"]["params"]["log_std"])))
                print(f"[ppo] it={it} reward/step={mean_r:.4f} "
                      f"track_lin={track_lin:.3f} track_yaw={track_yaw:.3f} "
                      f"pen={pen_mag:.3f} ep_len~{ep_len:.0f} "
                      f"ent={float(entropy):.2f} std={std:.3f} "
                      f"loss={float(loss):.4f} penalty_k={penalty_scale:.3f} "
                      f"steps/s={sps:.0f}")
                if tb is not None:
                    tb.add_scalar("train/entropy", float(entropy), it)
                    tb.add_scalar("policy/std", std, it)
                    tb.add_scalar("reward/step", mean_r, it)
                    tb.add_scalar("reward/tracking_lin", track_lin, it)
                    tb.add_scalar("reward/tracking_yaw", track_yaw, it)
                    tb.add_scalar("reward/penalty_mag", pen_mag, it)
                    tb.add_scalar("episode/done_rate", done_rate, it)
                    tb.add_scalar("episode/length_est", ep_len, it)
                    tb.add_scalar("train/loss", float(loss), it)
                    tb.add_scalar("train/penalty_k", penalty_scale, it)
                    tb.add_scalar("train/steps_per_s", sps, it)

            if it % p.save_every == 0 and it > 0:
                save_pytree(f"{self.cfg.checkpoint_dir}/phase1_{it}.pkl", params)

        save_pytree(f"{self.cfg.checkpoint_dir}/phase1_final.pkl", params)
        if tb is not None:
            tb.close()
        return params
