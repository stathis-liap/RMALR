"""Project-3 core figures: goal-conditioned velocity tracking.

These answer the central Project-3 question directly -- "how well does the Go2
follow the commanded target velocity (vx, vy, yaw), and does it adapt smoothly
when the command changes?":

  fig_tracking_timeseries  commanded vs achieved (vx, vy, yaw) through a scripted
                           command sequence, all three modes -> step response +
                           smooth command transitions.
  fig_tracking_accuracy    achieved vs commanded scatter over a command sweep,
                           with slope / R^2 / in-tolerance %  -> tracking accuracy.
  fig_tracking_polar       omnidirectional tracking in the vx-vy plane (commanded
                           circle vs achieved) -> the policy tracks every heading.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from . import core
from .core import MODES, MODE_COLOR, MODE_LABEL, MODE_LABEL_SHORT


# ---------------------------------------------------------------------------
def _scripted_schedule(T, N):
    """A held-segment command sequence shared across all N envs: (T,N,3).

    Walks through forward / stop / lateral / yaw / diagonal / turn-while-walking
    so the plot shows both steady tracking and command transitions.
    """
    segs = [
        (0.00, 0.0, 0.0),   # stand
        (0.6, 0.0, 0.0),    # forward
        (0.0, 0.0, 0.0),    # stop
        (0.0, 0.5, 0.0),    # strafe
        (0.0, 0.0, 0.7),    # turn in place
        (0.45, 0.0, 0.45),  # walk + turn
        (0.4, 0.4, 0.0),    # diagonal
        (0.0, 0.0, 0.0),    # stop
    ]
    seg_len = T // len(segs)
    cmd = np.zeros((T, 3), np.float32)
    for i, s in enumerate(segs):
        cmd[i * seg_len:(i + 1) * seg_len] = s
    cmd[len(segs) * seg_len:] = segs[-1]
    return np.broadcast_to(cmd[:, None, :], (T, N, 3)).copy()


def fig_tracking_timeseries(eng, out, N, T, seed=0, modes=MODES, warmup=75,
                            refresh=False):
    """Commanded vs achieved through a scripted sequence.

    A ``warmup`` of forward walking precedes the recorded sequence so the RMA
    adaptation module's 50-step history is filled before we measure (at a cold
    reset phi has no history and its z is unreliable for ~1 s). Fallen envs are
    masked out: a robot on the ground has a meaningless "achieved velocity", so
    the mean/std are taken over the still-walking envs only.
    """
    cache = core.cache_path("tracking_timeseries")
    if not refresh and __import__("os").path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        cmd, ach, alive = d["cmd"], d["ach"].item(), d["alive"].item()
    else:
        scripted = _scripted_schedule(T, N)                 # (T,N,3)
        warm = np.broadcast_to(np.array([0.4, 0.0, 0.0], np.float32),
                               (warmup, N, 3))
        sched = np.concatenate([warm, scripted], 0)         # (warmup+T, N, 3)
        f = core.nominal_factors(N)
        ach, alive = {}, {}
        for m in modes:
            with core.timer(f"timeseries[{m}]"):
                r = eng.rollout(m, warmup + T, f["friction"], f["payload"],
                                f["com_xy"], f["motor"], sched, seed=seed)
            ach[m] = r.achieved[warmup:]                     # (T,N,3) post-warmup
            alive[m] = r.alive[warmup:]                      # (T,N) bool
        cmd = scripted[:, 0, :]                              # shared command (T,3)
        np.savez(cache, cmd=cmd, ach=ach, alive=alive)

    dt = eng.env.dt
    T = cmd.shape[0]
    t = np.arange(T) * dt
    chans = [("forward  vx", 0, "m/s", 1.3), ("lateral  vy", 1, "m/s", 1.3),
             ("yaw rate  ωz", 2, "rad/s", 1.8)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.8), sharex=True)
    for ax, (name, c, unit, ylim) in zip(axes, chans):
        ax.plot(t, cmd[:, c], "k--", lw=2.2, label="commanded", zorder=5)
        for m in modes:
            a = np.where(alive[m], ach[m][:, :, c], np.nan)  # drop fallen envs
            with np.errstate(invalid="ignore"):
                mu, sd = np.nanmean(a, 1), np.nanstd(a, 1)
            ax.plot(t, mu, color=MODE_COLOR[m], lw=1.8, label=MODE_LABEL_SHORT[m])
            ax.fill_between(t, mu - sd, mu + sd, color=MODE_COLOR[m], alpha=0.15)
        ax.set_ylabel(f"{name}\n({unit})")
        ax.set_ylim(-ylim, ylim)
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=4, loc="upper center", fontsize=9, framealpha=0.9,
                   bbox_to_anchor=(0.5, 1.02))
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Goal-conditioned velocity tracking: commanded vs achieved "
                 f"(mean ± std over walking envs, n={N})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def _settled_mean(achieved, frac=0.5):
    """Mean of the achieved signal over the settled (last `frac`) of the episode."""
    T = achieved.shape[0]
    return achieved[int(T * (1 - frac)):].mean(0)        # (N,3)


def fig_tracking_accuracy(eng, out, N, T, seed=0, mode="rma", refresh=False):
    cache = core.cache_path("tracking_accuracy")
    if not refresh and __import__("os").path.exists(cache):
        d = np.load(cache)
        cmd_all, ach_all = d["cmd"], d["ach"]
    else:
        chans = {0: (-0.8, 0.8), 1: (-0.6, 0.6), 2: (-0.9, 0.9)}
        cmd_all, ach_all = [], []
        for c, (lo, hi) in chans.items():
            vals = np.linspace(lo, hi, N).astype(np.float32)
            cmds = np.zeros((N, 3), np.float32)
            cmds[:, c] = vals
            sched = core.const_commands(cmds, T)
            f = core.nominal_factors(N)
            with core.timer(f"accuracy[ch{c}]"):
                r = eng.rollout(mode, T, f["friction"], f["payload"], f["com_xy"],
                                f["motor"], sched, seed=seed)
            survived = r.survival()
            ach = _settled_mean(r.achieved)
            row_c = np.full(N, np.nan); row_a = np.full(N, np.nan)
            row_c[survived] = vals[survived]
            row_a[survived] = ach[survived, c]
            cmd_all.append(row_c); ach_all.append(row_a)
        cmd_all = np.array(cmd_all); ach_all = np.array(ach_all)
        np.savez(cache, cmd=cmd_all, ach=ach_all)

    names = ["forward  vx (m/s)", "lateral  vy (m/s)", "yaw rate  ωz (rad/s)"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for c, ax in enumerate(axes):
        cx, ay = cmd_all[c], ach_all[c]
        ok = np.isfinite(cx) & np.isfinite(ay)
        cx, ay = cx[ok], ay[ok]
        lim = max(np.abs(cx).max(), np.abs(ay).max(), 0.1) * 1.15
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2, label="ideal", zorder=1)
        ax.scatter(cx, ay, s=26, color=MODE_COLOR[mode], alpha=0.8, zorder=3)
        if len(cx) > 2:
            slope, intercept = np.polyfit(cx, ay, 1)
            ss_res = np.sum((ay - (slope * cx + intercept)) ** 2)
            ss_tot = np.sum((ay - ay.mean()) ** 2) + 1e-9
            r2 = 1 - ss_res / ss_tot
            tol = 0.15 if c < 2 else 0.2
            within = np.mean(np.abs(ay - cx) <= tol) * 100
            ax.text(0.04, 0.96, f"slope={slope:.2f}\n$R^2$={r2:.2f}\n"
                    f"≤{tol:g}: {within:.0f}%", transform=ax.transAxes,
                    va="top", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.set_xlabel("commanded"); ax.set_ylabel("achieved (settled)")
        ax.set_title(names[c]); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.suptitle(f"Velocity-tracking accuracy — {MODE_LABEL[mode]}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_tracking_polar(eng, out, N, T, seed=0, modes=("no_adapt", "rma"),
                       speed=0.5, refresh=False):
    cache = core.cache_path("tracking_polar")
    if not refresh and __import__("os").path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        cmd_xy, ach = d["cmd_xy"], d["ach"].item()
    else:
        head = np.linspace(-np.pi, np.pi, N, endpoint=False).astype(np.float32)
        cmds = np.stack([speed * np.cos(head), speed * np.sin(head),
                         np.zeros(N)], axis=1).astype(np.float32)
        sched = core.const_commands(cmds, T)
        f = core.nominal_factors(N)
        ach = {}
        for m in modes:
            with core.timer(f"polar[{m}]"):
                r = eng.rollout(m, T, f["friction"], f["payload"], f["com_xy"],
                                f["motor"], sched, seed=seed)
            a = _settled_mean(r.achieved)[:, :2]
            a[~r.survival()] = np.nan
            ach[m] = a
        cmd_xy = cmds[:, :2]
        np.savez(cache, cmd_xy=cmd_xy, ach=ach)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(speed * np.cos(th), speed * np.sin(th), "k--", lw=1.3,
            label=f"commanded |v|={speed} m/s")
    ax.scatter(cmd_xy[:, 0], cmd_xy[:, 1], s=18, c="k", alpha=0.5, zorder=2)
    for m in modes:
        a = ach[m]
        ax.scatter(a[:, 0], a[:, 1], s=34, color=MODE_COLOR[m],
                   label=MODE_LABEL_SHORT[m], zorder=4, alpha=0.85)
        for i in range(len(a)):
            if np.isfinite(a[i, 0]):
                ax.plot([cmd_xy[i, 0], a[i, 0]], [cmd_xy[i, 1], a[i, 1]],
                        color=MODE_COLOR[m], lw=0.5, alpha=0.4, zorder=3)
    lim = speed * 1.6
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("vx achieved (m/s)"); ax.set_ylabel("vy achieved (m/s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Omnidirectional velocity tracking\n(commanded heading circle "
                 "vs achieved planar velocity)", fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
