"""Adaptation-module figures: does phi actually recover the extrinsics?

This is the part the paper argues is its real contribution (Sec III-B, Figs 4 / S3):
the adaptation module phi estimates the latent extrinsics z online from the
recent state/action history, with no access to the privileged e_t.

  fig_latent_embedding   PCA of z: phi(history) recovers the same latent structure
                         as the privileged mu(e_t), and the two agree (expansion
                         on the paper -- it never visualizes the latent space).
  fig_adaptation         the extrinsics estimate responding to the environment:
                         (a,b) phi converging to mu's target as history fills for
                         light/heavy payload and low/high friction, and (c) phi
                         tracking a *mid-episode* actuator degradation -- the
                         paper's Fig-4 "disturbance -> z shifts -> gait recovers".
"""
from __future__ import annotations

import os

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from . import core
from .core import MODE_COLOR


# ---------------------------------------------------------------------------
def fig_latent_embedding(eng, out, N, T, seed=0, refresh=False):
    cache = core.cache_path("embedding")
    if not refresh and os.path.exists(cache):
        d = np.load(cache)
        z_rma, e = d["z_rma"], d["e"]
    else:
        # wide factor spread (train range and a bit beyond) so structure is visible
        rng = np.random.default_rng(seed)
        friction = rng.uniform(0.15, 2.4, N).astype(np.float32)
        payload = rng.uniform(0.0, 11.0, N).astype(np.float32)
        com_xy = rng.uniform(-0.1, 0.1, (N, 2)).astype(np.float32)
        motor = rng.uniform(0.6, 1.15, N).astype(np.float32)
        cmd = core.const_commands(core.sample_commands(N, seed=seed + 1), T)
        with core.timer("embedding rollout"):
            r = eng.rollout("rma", T, friction, payload, com_xy, motor, cmd, seed=seed)
        # settled phi estimate: mean of z over the last alive steps per env
        last = max(20, T // 6)
        zt = r.z[-last:]                                # (last, N, 8)
        with np.errstate(invalid="ignore"):
            z_rma = np.nanmean(zt, axis=0)             # (N, 8)
        e = r.e                                         # (N, 17)
        np.savez(cache, z_rma=z_rma, e=e)

    # drop any env that diverged to NaN (extreme OOD payload/motor) before PCA
    keep = np.isfinite(z_rma).all(1) & np.isfinite(e).all(1)
    z_rma, e = z_rma[keep], e[keep]
    z_mu = eng.mu(e)                                    # (N, 8) privileged target
    payload = e[:, 0]                                   # e layout: payload, com2, motor12, fric, terr
    friction = e[:, 15]

    proj_mu, basis = core.pca_2d(z_mu)
    proj_phi, _ = core.pca_2d(z_rma, basis=basis)       # shared axes

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    sc = axes[0].scatter(proj_mu[:, 0], proj_mu[:, 1], c=payload, cmap="viridis", s=26)
    axes[0].set_title("μ(eₜ) — privileged encoder")
    axes[1].scatter(proj_phi[:, 0], proj_phi[:, 1], c=payload, cmap="viridis", s=26)
    axes[1].set_title("φ(history) — adaptation module")
    for ax in axes[:2]:
        ax.set_xlabel("z PCA-1"); ax.set_ylabel("z PCA-2"); ax.grid(alpha=0.3)
    cb = fig.colorbar(sc, ax=axes[1]); cb.set_label("payload (kg)")

    # agreement: PCA-1 of phi vs PCA-1 of mu (shared basis) -> recovery quality
    a, b = proj_mu[:, 0], proj_phi[:, 0]
    lim = max(np.abs(a).max(), np.abs(b).max()) * 1.1
    axes[2].plot([-lim, lim], [-lim, lim], "k--", lw=1.1)
    axes[2].scatter(a, b, s=22, c=friction, cmap="plasma")
    ss_res = np.sum((b - a) ** 2); ss_tot = np.sum((b - b.mean()) ** 2) + 1e-9
    corr = np.corrcoef(a, b)[0, 1]
    axes[2].text(0.04, 0.96, f"corr={corr:.2f}", transform=axes[2].transAxes,
                 va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    axes[2].set_xlabel("μ(eₜ)  PCA-1"); axes[2].set_ylabel("φ(history)  PCA-1")
    axes[2].set_title("φ vs μ agreement (color = friction)")
    axes[2].set_aspect("equal"); axes[2].grid(alpha=0.3)
    fig.suptitle("Latent extrinsics: φ recovers the privileged encoder's structure "
                 "from history alone", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def _mu_target(eng, payload=0.0, com=(0.0, 0.0), motor=1.0, friction=1.0, terr=0.0):
    """mu(e) for a hand-built privileged vector (training e_t layout)."""
    e = np.concatenate([[payload], com, np.full(12, motor), [friction], [terr]])
    return eng.mu(e.astype(np.float32))


def _proj_coord(z, t_lo, t_hi):
    """Signed coordinate of z along the (t_hi - t_lo) direction, centred at midpoint."""
    d = t_hi - t_lo
    d = d / (np.linalg.norm(d) + 1e-9)
    mid = 0.5 * (t_lo + t_hi)
    return (z - mid) @ d, (t_lo - mid) @ d, (t_hi - mid) @ d


def _motor_event_rollout(eng, T, event_t, m_before, m_after, N, seed):
    """rma rollout, nominal except motor strength stepped m_before->m_after at event_t.

    phi must infer the actuator change from the altered dynamics in its history.
    """
    env = eng.env
    fr = np.ones(N, np.float32)
    batched, in_axes = core._build_batched(
        env, fr, np.zeros(N, np.float32), np.zeros((N, 2), np.float32))
    v_reset = jax.vmap(env.reset, in_axes=(in_axes, 0))
    v_step = jax.vmap(env.step, in_axes=(in_axes, 0, 0, None))
    p1, pol, adapt, phi = eng._p1, eng._pol, eng._adapt, eng._phi
    cmd = jnp.asarray(core.const_commands(
        core.sample_commands(N, seed=seed + 2), T))
    mb = jnp.full((N, 12), m_before, jnp.float32)
    ma = jnp.full((N, 12), m_after, jnp.float32)

    @jax.jit
    def run(batched, state):
        def body(carry, inp):
            state, z = carry
            cmd_t, t = inp
            state = state.replace(
                motor_strength=jnp.where(t >= event_t, ma, mb))
            z = jnp.where((t % 10) == 0, adapt.apply(phi, state.history), z)
            mean, _ = pol.apply(p1["policy"], state.obs, state.prev_action, z)
            state = state.replace(command=cmd_t)
            state = v_step(batched, state, mean, 1.0)
            lin = jnp.linalg.norm(
                state.obs[:, core.OBS_SLICES["lin_vel_err"]][:, :2], axis=-1)
            return (state, z), (z, lin, state.done)
        z0 = adapt.apply(phi, state.history)
        _, out = jax.lax.scan(body, (state, z0), (cmd, jnp.arange(T)))
        return out

    state = v_reset(batched, jax.random.split(jax.random.PRNGKey(seed), N))
    state = state.replace(motor_strength=mb)
    z, lin, done = (np.asarray(x) for x in jax.block_until_ready(run(batched, state)))
    alive = np.cumsum(done > 0.5, 0) == 0
    return z, np.nan_to_num(lin), alive


def fig_adaptation(eng, out, N, T, seed=0, refresh=False):
    cache = core.cache_path("adaptation")
    if not refresh and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        conv = d["conv"].item(); ev = d["ev"].item()
    else:
        # --- (a,b) convergence: 4 conditions split across the batch -----------
        g = N // 4
        payload = np.zeros(4 * g, np.float32)
        friction = np.ones(4 * g, np.float32)
        motor = np.ones(4 * g, np.float32)
        payload[g:2 * g] = 10.0                       # heavy group
        friction[2 * g:3 * g] = 0.2                   # low-friction group
        friction[3 * g:4 * g] = 2.2                   # high-friction group
        com = np.zeros((4 * g, 2), np.float32)
        cmd = core.const_commands(core.sample_commands(4 * g, seed=seed + 3), T)
        with core.timer("adaptation convergence"):
            r = eng.rollout("rma", T, friction, payload, com, motor, cmd, seed=seed)
        groups = {"light": slice(0, g), "heavy": slice(g, 2 * g),
                  "lowfric": slice(2 * g, 3 * g), "highfric": slice(3 * g, 4 * g)}
        conv = {k: np.nanmean(r.z[:, sl, :], axis=1) for k, sl in groups.items()}

        # --- (c) mid-episode actuator degradation event ----------------------
        with core.timer("adaptation event"):
            z, lin, alive = _motor_event_rollout(
                eng, T, event_t=T // 3, m_before=1.0, m_after=0.6,
                N=max(16, N // 2), seed=seed)
        ev = dict(z=z.mean(1), lin=(lin * alive).sum(1) / np.maximum(alive.sum(1), 1),
                  event_t=T // 3)
        np.savez(cache, conv=conv, ev=ev)

    dt = eng.env.dt
    T = conv["light"].shape[0]                          # honour the cached horizon
    t = np.arange(T) * dt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) payload
    t_light = _mu_target(eng, payload=0.0)
    t_heavy = _mu_target(eng, payload=10.0)
    pl, plo, phi_ = _proj_coord(conv["light"], t_light, t_heavy)
    ph, _, _ = _proj_coord(conv["heavy"], t_light, t_heavy)
    axes[0].plot(t, pl, color="#7aa6c2", lw=2, label="0 kg")
    axes[0].plot(t, ph, color="#0b3d5c", lw=2, label="+10 kg")
    axes[0].axhline(plo, ls=":", color="#7aa6c2"); axes[0].axhline(phi_, ls=":", color="#0b3d5c")
    axes[0].set_title("payload adaptation"); axes[0].legend(fontsize=9)

    # (b) friction
    t_lo = _mu_target(eng, friction=0.2)
    t_hi = _mu_target(eng, friction=2.2)
    fl, flo, fhi = _proj_coord(conv["lowfric"], t_lo, t_hi)
    fh, _, _ = _proj_coord(conv["highfric"], t_lo, t_hi)
    axes[1].plot(t, fl, color="#e0a050", lw=2, label="friction 0.2")
    axes[1].plot(t, fh, color="#8a4b00", lw=2, label="friction 2.2")
    axes[1].axhline(flo, ls=":", color="#e0a050"); axes[1].axhline(fhi, ls=":", color="#8a4b00")
    axes[1].set_title("friction adaptation"); axes[1].legend(fontsize=9)

    for ax in axes[:2]:
        ax.set_xlabel("time (s)"); ax.set_ylabel("extrinsics ẑ coordinate")
        ax.grid(alpha=0.3)
        ax.text(0.98, 0.5, "μ(eₜ) targets (dotted)", transform=ax.transAxes,
                ha="right", va="center", fontsize=8, color="0.4")

    # (c) actuator-degradation event
    te = np.arange(ev["z"].shape[0]) * dt
    t_full = _mu_target(eng, motor=1.0); t_weak = _mu_target(eng, motor=0.6)
    zc, zlo, zhi = _proj_coord(ev["z"], t_full, t_weak)
    ax = axes[2]
    ax.plot(te, zc, color=MODE_COLOR["rma"], lw=2, label="φ ẑ (motor coord)")
    ax.axhline(zlo, ls=":", color="0.4"); ax.axhline(zhi, ls=":", color="0.4")
    ax.axvline(ev["event_t"] * dt, color="red", lw=1.4, ls="--")
    ax.text(ev["event_t"] * dt, ax.get_ylim()[1], " motor→0.6×", color="red",
            fontsize=8, va="top")
    ax2 = ax.twinx()
    ax2.plot(te, ev["lin"], color="0.5", lw=1.2, alpha=0.8)
    ax2.set_ylabel("lin-err (m/s)", color="0.5")
    ax.set_xlabel("time (s)"); ax.set_ylabel("extrinsics ẑ coordinate")
    ax.set_title("mid-episode actuator degradation"); ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.suptitle("Online adaptation: φ’s extrinsics estimate tracks the "
                 "environment from history", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
