"""Small shared utilities: checkpoint I/O and pytree selection."""
from __future__ import annotations

import os
import pickle

import jax
import jax.numpy as jnp


def save_pytree(path: str, tree) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flat = jax.device_get(tree)
    with open(path, "wb") as f:
        pickle.dump(flat, f)


def load_pytree(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def tree_select(mask, a, b):
    """Per-leaf jnp.where(mask, a, b); mask broadcasts over leading env axis."""
    def sel(x, y):
        m = mask.reshape(mask.shape + (1,) * (x.ndim - mask.ndim))
        return jnp.where(m, x, y)
    return jax.tree_util.tree_map(sel, a, b)
