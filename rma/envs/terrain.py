"""Fractal (fBm value-noise) heightfield, as in the paper's RaiSim generator.

Returns a normalized [0,1] field; the caller scales it to metres via the MuJoCo
hfield asset's elevation range. One field is shared across the MJX batch, so
per-env terrain variety comes from randomized spawn positions, not the field.
"""
from __future__ import annotations

import numpy as np


def _smooth(t):
    return t * t * (3 - 2 * t)


def _sample_grid(grid, res, out_shape):
    """Bilinearly upsample a coarse `grid` (res x res) to `out_shape`, smoothed."""
    h, w = out_shape
    ys = np.linspace(0, res - 1, h)
    xs = np.linspace(0, res - 1, w)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.clip(y0 + 1, 0, res - 1), np.clip(x0 + 1, 0, res - 1)
    ty, tx = _smooth(ys - y0)[:, None], _smooth(xs - x0)[None, :]
    top = grid[np.ix_(y0, x0)] * (1 - tx) + grid[np.ix_(y0, x1)] * tx
    bot = grid[np.ix_(y1, x0)] * (1 - tx) + grid[np.ix_(y1, x1)] * tx
    return top * (1 - ty) + bot * ty


def fractal_heightfield(
    size: int = 256,
    octaves: int = 2,
    lacunarity: float = 2.0,
    gain: float = 0.25,
    base_frequency: int = 16,
    seed: int = 0,
) -> np.ndarray:
    """Return an (size x size) float32 fBm field normalized to [0, 1]."""
    rng = np.random.default_rng(seed)
    field = np.zeros((size, size), dtype=np.float64)
    amplitude, frequency, total_amp = 1.0, float(base_frequency), 0.0
    for _ in range(octaves):
        res = max(2, int(frequency))
        field += amplitude * _sample_grid(rng.random((res, res)), res, (size, size))
        total_amp += amplitude
        amplitude *= gain
        frequency *= lacunarity
    field = (field / max(total_amp, 1e-8))
    field -= field.min()
    field /= max(field.max(), 1e-8)
    return field.astype(np.float32)
