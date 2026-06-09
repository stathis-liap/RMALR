"""Flax networks for RMA (paper Sec IV-B).

* EnvFactorEncoder (mu): e_t (17) -> z_t (8)
* BasePolicy (pi):      [x_t (30), a_{t-1} (12), z_t (8)] -> action mean (12)
* ValueNet (asymmetric critic): [x_t, a_{t-1}, e_t] -> scalar value
* AdaptationModule (phi): history (k, 42) -> z_hat (8) via MLP embed + 1-D CNN
"""
from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


def mlp(x, hidden: Sequence[int], out: int, activation=nn.elu, name_prefix="fc"):
    for i, h in enumerate(hidden):
        x = nn.Dense(h, name=f"{name_prefix}{i}")(x)
        x = activation(x)
    return nn.Dense(out, name=f"{name_prefix}_out")(x)


class EnvFactorEncoder(nn.Module):
    hidden: Tuple[int, ...]
    latent_dim: int

    @nn.compact
    def __call__(self, e):
        return mlp(e, self.hidden, self.latent_dim, name_prefix="enc")


class BasePolicy(nn.Module):
    hidden: Tuple[int, ...]
    action_dim: int
    log_std_init: float
    min_log_std: float

    @nn.compact
    def __call__(self, x, a_prev, z):
        inp = jnp.concatenate([x, a_prev, z], axis=-1)
        mean = mlp(inp, self.hidden, self.action_dim, name_prefix="pi")
        log_std = self.param("log_std", nn.initializers.constant(self.log_std_init),
                             (self.action_dim,))
        log_std = jnp.maximum(log_std, self.min_log_std)
        return mean, log_std


class ValueNet(nn.Module):
    hidden: Tuple[int, ...]

    @nn.compact
    def __call__(self, x, a_prev, e):
        inp = jnp.concatenate([x, a_prev, e], axis=-1)
        v = mlp(inp, self.hidden, 1, name_prefix="vf")
        return jnp.squeeze(v, -1)


class AdaptationModule(nn.Module):
    embed_hidden: Tuple[int, ...]
    embed_dim: int
    conv: Tuple[Tuple[int, int, int, int], ...]
    latent_dim: int

    @nn.compact
    def __call__(self, history):
        # history: (..., k, 42). Embed each timestep to embed_dim.
        h = history
        for i, units in enumerate(self.embed_hidden):
            h = nn.elu(nn.Dense(units, name=f"emb{i}")(h))
        h = nn.Dense(self.embed_dim, name="emb_out")(h)   # (..., k, embed_dim)

        # 1-D CNN over the time axis. Flax Conv expects (..., length, channels).
        for i, (_in, out, k, s) in enumerate(self.conv):
            h = nn.Conv(features=out, kernel_size=(k,), strides=(s,),
                        padding="VALID", name=f"conv{i}")(h)
            h = nn.elu(h)

        h = h.reshape(h.shape[:-2] + (-1,))               # flatten time*channels
        z = nn.Dense(self.latent_dim, name="proj")(h)
        return z


# --------------------------------------------------------------------------
# Convenience builders + a small container for Phase-1 params.
# --------------------------------------------------------------------------
def build_networks(cfg):
    nc = cfg.net
    encoder = EnvFactorEncoder(hidden=nc.encoder_hidden, latent_dim=nc.latent_dim)
    policy = BasePolicy(hidden=nc.policy_hidden, action_dim=nc.action_dim,
                        log_std_init=nc.log_std_init, min_log_std=nc.min_log_std)
    value = ValueNet(hidden=nc.value_hidden)
    adapt = AdaptationModule(embed_hidden=nc.adapt_embed_hidden,
                             embed_dim=nc.adapt_embed_dim, conv=nc.adapt_conv,
                             latent_dim=nc.latent_dim)
    return encoder, policy, value, adapt


def gaussian_log_prob(mean, log_std, action):
    std = jnp.exp(log_std)
    pre = -0.5 * (((action - mean) / std) ** 2) - log_std - 0.5 * jnp.log(2 * jnp.pi)
    return jnp.sum(pre, axis=-1)


def gaussian_entropy(log_std):
    return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
