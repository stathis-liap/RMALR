"""Fractal terrain generation (paper uses RaiSim's fractal generator).

RaiSim parameters: octaves=2, lacunarity=2.0, gain=0.25, z-scale=0.27.
We reproduce the height field with value-noise fBm and expose it as a numpy
array suitable for a MuJoCo heightfield asset.

NOTE on MJX: per-environment heightfields are not supported (the model is shared
across the vmapped batch). We therefore use ONE shared fractal field and let the
randomly-spawned robots sample different local terrain. Heightfield <-> geom
collision support in MJX also varies by mujoco version; `terrain="flat"` (the
default in config) sidesteps both issues and always runs.
"""
from __future__ import annotations

import numpy as np


def _value_noise(shape, rng):
    return rng.random(shape, dtype=np.float64)


def _lerp(a, b, t):
    return a + (b - a) * t


def _smooth(t):
    return t * t * (3 - 2 * t)


def _sample_grid(grid, res, out_shape):
    """Bilinearly upsample a coarse `grid` (res x res) to `out_shape`."""
    h, w = out_shape
    ys = np.linspace(0, res - 1, h)
    xs = np.linspace(0, res - 1, w)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, res - 1)
    x1 = np.clip(x0 + 1, 0, res - 1)
    ty = _smooth(ys - y0)[:, None]
    tx = _smooth(xs - x0)[None, :]
    g = grid
    top = _lerp(g[np.ix_(y0, x0)], g[np.ix_(y0, x1)], tx)
    bot = _lerp(g[np.ix_(y1, x0)], g[np.ix_(y1, x1)], tx)
    return _lerp(top, bot, ty)


def fractal_heightfield(
    size: int = 256,
    octaves: int = 2,
    lacunarity: float = 2.0,
    gain: float = 0.25,
    z_scale: float = 0.27,
    seed: int = 0,
) -> np.ndarray:
    """Return an (size x size) float32 height map normalized to [0, 1].

    `z_scale` is applied by the caller via the heightfield asset's elevation
    range, so here we just return a normalized fBm field.
    """
    rng = np.random.default_rng(seed)
    field = np.zeros((size, size), dtype=np.float64)
    amplitude = 1.0
    frequency = 4
    total_amp = 0.0
    for _ in range(octaves):
        res = max(2, int(frequency))
        grid = _value_noise((res, res), rng)
        field += amplitude * _sample_grid(grid, res, (size, size))
        total_amp += amplitude
        amplitude *= gain
        frequency *= lacunarity
    field /= max(total_amp, 1e-8)
    field -= field.min()
    field /= max(field.max(), 1e-8)
    return field.astype(np.float32)
