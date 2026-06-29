"""Robustness / baseline-comparison figures (the RMA paper's Fig 3 / S4 / Table II).

Bracket the policy between its lower bound (no-adapt) and upper bound (expert),
exactly like the paper's Robust / RMA / Expert comparison, across the
perturbations the Project-3 hidden tests name: friction, payload/mass, actuator
strength, COM, and external pushes.

  fig_generalization_sweeps   survival + tracking error vs friction / payload /
                              motor strength, 3 modes  (paper Fig S4).
  fig_stress_bars             survival + tracking error per stress scenario,
                              3 modes  (paper Fig 3).
  summary_table               aggregate metrics over the full DR distribution,
                              3 modes  (paper Table II)  -> csv + md + png.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib.pyplot as plt

from . import core
from .core import MODES, MODE_COLOR, MODE_LABEL, MODE_LABEL_SHORT

# training ranges (rma/config.py) -> shaded "in-distribution" spans on the sweeps
TRAIN_RANGE = {"friction": (0.30, 2.0), "payload": (0.0, 5.0), "motor": (0.85, 1.15)}
SWEEP = {
    "friction": ("ground friction", (0.1, 2.6), "friction"),
    "payload":  ("trunk payload (kg)", (0.0, 12.0), "payload"),
    "motor":    ("motor strength ×", (0.5, 1.2), "motor"),
}


def _binned(x, y, edges, reduce=np.mean):
    """Reduce y in each [edges[i], edges[i+1]) bin; returns (centers, vals, counts).

    survival uses mean (-> fraction); tracking error uses median so a single
    near-fall env that diverges before tipping doesn't dominate a sparse bin.
    """
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    centers, vals, counts = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        vals.append(reduce(y[m]) if m.any() else np.nan)
        counts.append(int(m.sum()))
    return np.array(centers), np.array(vals), np.array(counts)


def fig_generalization_sweeps(eng, out, N, T, seed=0, modes=MODES, nbins=7,
                              refresh=False):
    cache = core.cache_path("generalization")
    if not refresh and os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)["data"].item()
    else:
        data = {}
        cmd = core.const_commands(core.sample_commands(N, seed=seed), T)
        for axis, (_, (lo, hi), key) in SWEEP.items():
            grid = np.linspace(lo, hi, N).astype(np.float32)
            f = core.nominal_factors(N)
            f[key] = grid                         # overwrite the swept axis
            for m in modes:
                with core.timer(f"sweep[{axis},{m}]"):
                    r = eng.rollout(m, T, f["friction"], f["payload"],
                                    f["com_xy"], f["motor"], cmd, seed=seed)
                data[(axis, m)] = dict(x=grid, surv=r.survival().astype(float),
                                       lin=r.ep_lin_err())
        np.savez(cache, data=data)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for j, (axis, (xlabel, (lo, hi), _)) in enumerate(SWEEP.items()):
        edges = np.linspace(lo, hi, nbins + 1)
        tr = TRAIN_RANGE[axis]
        for row, metric, ylab in ((0, "surv", "survival"), (1, "lin", "lin-err (m/s)")):
            ax = axes[row, j]
            ax.axvspan(tr[0], tr[1], color="0.85", alpha=0.6, zorder=0,
                       label="train range" if (row == 0 and j == 0) else None)
            for m in modes:
                d = data[(axis, m)]
                reducer = np.mean if metric == "surv" else np.median
                c, val, _ = _binned(d["x"], d[metric], edges, reduce=reducer)
                if metric == "surv":
                    val = val * 100
                ax.plot(c, val, "-o", ms=4, color=MODE_COLOR[m], lw=1.8,
                        label=MODE_LABEL_SHORT[m] if (row == 0 and j == 0) else None)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_ylim(-3, 103); ax.set_title(xlabel.split(" (")[0])
            ax.set_ylabel(ylab if j == 0 else "")
            if row == 1:
                ax.set_xlabel(xlabel)
    axes[0, 0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Generalization to out-of-distribution physics "
                 "(survival ↑ and tracking error ↓ vs perturbation)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stress scenarios (reuse testing/scenarios.py presets, mapped onto the MJX env)
# ---------------------------------------------------------------------------
def _scenario_arrays(sc, N):
    """Per-env factor arrays + push for a Scenario (constant across the batch)."""
    fr = 0.5 * (sc.friction[0] + sc.friction[1])
    return dict(
        friction=np.full(N, fr, np.float32),
        payload=np.full(N, sc.payload, np.float32),
        com_xy=np.tile([sc.com_shift[0], sc.com_shift[1]], (N, 1)).astype(np.float32),
        motor=np.full(N, sc.motor_scale, np.float32),
        push_mag=(np.full(N, sc.push_vel, np.float32) if sc.push_interval else None),
        push_period=(sc.push_interval or 0),
    )


# curated subset for the bar chart. Each isolates one physics stressor on flat
# ground (the continuous axes are covered by the sweeps; these add the push / COM
# scenarios). Combined / rough-terrain-scene presets (gauntlet, perlin, boxes)
# are deliberately excluded -- on gym-quadruped's over-tall terrain the legs clip
# through, so those runs are not indicative; rough terrain is tested faithfully on
# the MJX training heightfield instead (see fig_hfield_sweep).
STRESS_SET = ["nominal", "slippery", "ice", "payload_5kg", "payload_10kg",
              "front_heavy", "very_weak_motors", "pushes_hard"]


def fig_stress_bars(eng, out, N, T, seed=0, modes=MODES, names=None, refresh=False):
    from testing.scenarios import SCENARIOS
    names = names or STRESS_SET
    cache = core.cache_path("stress_bars")
    if not refresh and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        rows = d["rows"].item()
    else:
        cmd = core.const_commands(core.sample_commands(N, seed=seed), T)
        rows = {}
        for name in names:
            a = _scenario_arrays(SCENARIOS[name], N)
            for m in modes:
                with core.timer(f"stress[{name},{m}]"):
                    r = eng.rollout(m, T, a["friction"], a["payload"], a["com_xy"],
                                    a["motor"], cmd, seed=seed,
                                    push_mag=a["push_mag"], push_period=a["push_period"])
                rows[(name, m)] = dict(surv=r.survival().mean() * 100,
                                       lin=r.ep_lin_err().mean())
        np.savez(cache, rows=rows)

    x = np.arange(len(names)); w = 0.26
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    for i, m in enumerate(modes):
        off = (i - 1) * w
        a0.bar(x + off, [rows[(n, m)]["surv"] for n in names], w,
               color=MODE_COLOR[m], label=MODE_LABEL_SHORT[m])
        a1.bar(x + off, [rows[(n, m)]["lin"] for n in names], w,
               color=MODE_COLOR[m])
    a0.set_ylabel("survival (%)"); a0.set_ylim(0, 105); a0.legend(fontsize=9)
    a0.grid(axis="y", alpha=0.3); a1.grid(axis="y", alpha=0.3)
    a1.set_ylabel("lin-vel track err (m/s)")
    a1.set_xticks(x); a1.set_xticklabels(names, rotation=30, ha="right")
    fig.suptitle("Stress scenarios: lower bound (no-adapt) ≤ RMA ≤ upper bound "
                 "(expert)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table II analog: aggregate metrics over the full training DR distribution.
# ---------------------------------------------------------------------------
METRICS = [("survival", "%", "{:.1f}"), ("TTF", "", "{:.3f}"),
           ("reward/step", "", "{:.3f}"), ("lin_err", "m/s", "{:.3f}"),
           ("yaw_err", "rad/s", "{:.3f}"), ("torque", "N²m²", "{:.0f}"),
           ("smoothness", "", "{:.1f}")]


def summary_table(eng, out_png, N, T, seed=0, modes=MODES, refresh=False):
    cache = core.cache_path("summary")
    if not refresh and os.path.exists(cache):
        table = np.load(cache, allow_pickle=True)["table"].item()
    else:
        rng = np.random.default_rng(seed)
        ec = eng.cfg.env
        friction = rng.uniform(*ec.friction_range, N).astype(np.float32)
        payload = rng.uniform(*ec.payload_range, N).astype(np.float32)
        com_xy = rng.uniform(*ec.com_range, (N, 2)).astype(np.float32)
        motor = rng.uniform(*ec.motor_strength_range, N).astype(np.float32)
        cmd = core.const_commands(core.sample_commands(N, seed=seed + 5), T)
        table = {}
        for m in modes:
            with core.timer(f"summary[{m}]"):
                r = eng.rollout(m, T, friction, payload, com_xy, motor, cmd,
                                seed=seed)
            table[m] = dict(
                survival=r.survival().mean() * 100, TTF=r.ttf_norm().mean(),
                **{"reward/step": r.reward_per_step().mean()},
                lin_err=r.ep_lin_err().mean(), yaw_err=r.ep_yaw_err().mean(),
                torque=r.torque_sq().mean(), smoothness=r.smoothness().mean())
        np.savez(cache, table=table)

    # write csv + md next to the png
    base = os.path.splitext(out_png)[0]
    keys = [k for k, _, _ in METRICS]
    with open(base + ".csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["mode"] + keys)
        for m in modes:
            w.writerow([m] + [table[m][k] for k in keys])
    with open(base + ".md", "w") as fh:
        fh.write("| mode | " + " | ".join(
            f"{k}{(' ('+u+')') if u else ''}" for k, u, _ in METRICS) + " |\n")
        fh.write("|" + "---|" * (len(METRICS) + 1) + "\n")
        for m in modes:
            fh.write(f"| {MODE_LABEL_SHORT[m]} | " + " | ".join(
                f.format(table[m][k]) for k, _, f in METRICS) + " |\n")

    # rendered table figure
    fig, ax = plt.subplots(figsize=(11, 1.9))
    ax.axis("off")
    col = ["mode"] + [f"{k}\n({u})" if u else k for k, u, _ in METRICS]
    cells = [[MODE_LABEL_SHORT[m]] + [f.format(table[m][k])
             for k, _, f in METRICS] for m in modes]
    tb = ax.table(cellText=cells, colLabels=col, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 1.7)
    for j in range(len(col)):
        tb[0, j].set_facecolor("#30363d"); tb[0, j].set_text_props(color="w")
    for i, m in enumerate(modes):
        tb[i + 1, 0].set_facecolor(MODE_COLOR[m])
        tb[i + 1, 0].set_text_props(color="w")
    ax.set_title("Aggregate performance over the full domain-randomization "
                 "distribution  (paper Table II analog)", fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return table


# ---------------------------------------------------------------------------
# Rough terrain taken to the extreme -- on the FAITHFUL training heightfield.
# (gym-quadruped's perlin/boxes are over-tall and make the legs clip through,
#  so terrain robustness is measured on the MJX fractal heightfield instead.)
# ---------------------------------------------------------------------------
def fig_hfield_sweep(eng, out, N, T,
                     z_scales=(0.10, 0.30, 0.50, 0.70, 0.90, 1.20),
                     modes=MODES, seed=0, refresh=False):
    cache = core.cache_path("hfield_sweep")
    if not refresh and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        data = d["data"].item(); z_scales = d["z"]
    else:
        data = {m: dict(surv=[], lin=[]) for m in modes}
        for z in z_scales:
            heng = core.Engine(eng.ckpt1, eng.ckpt2, terrain="hfield", z_scale=float(z))
            f = core.nominal_factors(N)
            cmd = core.const_commands(core.sample_commands(N, seed=seed), T)
            for m in modes:
                with core.timer(f"hfield[z={z:.2f},{m}]"):
                    r = heng.rollout(m, T, f["friction"], f["payload"], f["com_xy"],
                                     f["motor"], cmd, seed=seed)
                data[m]["surv"].append(r.survival().mean() * 100)
                data[m]["lin"].append(np.median(r.ep_lin_err()))
        np.savez(cache, data=data, z=np.asarray(z_scales))

    z = np.asarray(z_scales)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.4))
    for m in modes:
        a0.plot(z, data[m]["surv"], "-o", color=MODE_COLOR[m], lw=1.8,
                label=MODE_LABEL_SHORT[m])
        a1.plot(z, data[m]["lin"], "-o", color=MODE_COLOR[m], lw=1.8)
    lin_max = max(0.16, 1.15 * max(max(data[m]["lin"]) for m in modes))
    for ax in (a0, a1):
        ax.axvspan(0.0, 0.10, color="0.88", alpha=0.7, zorder=0)   # <= training
        ax.set_xlabel("fractal terrain height z-scale (m)"); ax.grid(alpha=0.3)
    a0.text(0.11, 4, "train", color="0.4", fontsize=8)
    a0.set_ylabel("survival (%)"); a0.set_ylim(-3, 103); a0.legend(fontsize=9)
    a1.set_ylabel("lin-vel track err (m/s)"); a1.set_ylim(0, lin_max)
    fig.suptitle("Robustness to terrain roughness — MJX training heightfield, "
                 "swept to 1.2 m (12$\\times$ training)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Proprioceptive sensor noise (a named Project-3 hidden test) taken to extremes.
# ---------------------------------------------------------------------------
def fig_sensor_noise(eng, out, N, T, noises=(0.0, 0.02, 0.05, 0.1, 0.15, 0.2),
                     modes=MODES, seed=0, refresh=False):
    cache = core.cache_path("sensor_noise")
    if not refresh and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        data = d["data"].item(); noises = d["noises"]
    else:
        f = core.nominal_factors(N)
        cmd = core.const_commands(core.sample_commands(N, seed=seed), T)
        data = {m: dict(surv=[], lin=[]) for m in modes}
        for s in noises:
            for m in modes:
                with core.timer(f"noise[{s:.2f},{m}]"):
                    r = eng.rollout(m, T, f["friction"], f["payload"], f["com_xy"],
                                    f["motor"], cmd, seed=seed, obs_noise=float(s))
                data[m]["surv"].append(r.survival().mean() * 100)
                data[m]["lin"].append(np.median(r.ep_lin_err()))
        np.savez(cache, data=data, noises=np.asarray(noises))

    x = np.asarray(noises)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.4))
    for m in modes:
        a0.plot(x, data[m]["surv"], "-o", color=MODE_COLOR[m], lw=1.8,
                label=MODE_LABEL_SHORT[m])
        a1.plot(x, data[m]["lin"], "-o", color=MODE_COLOR[m], lw=1.8)
    a0.set_ylabel("survival (%)"); a0.set_ylim(-3, 103); a0.legend(fontsize=9)
    a1.set_ylabel("lin-vel track err (m/s)")
    for ax in (a0, a1):
        ax.set_xlabel("sensor-noise std on proprioceptive xₜ"); ax.grid(alpha=0.3)
    fig.suptitle("Robustness to proprioceptive sensor noise "
                 "(injected on the policy input)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Compound difficulty: ramp ALL strongly-privileged factors together. This is
# where knowing e_t matters most, so the no_adapt <= rma <= expert ordering
# separates cleanly (each axis is laid across the env batch -> one rollout/mode).
# ---------------------------------------------------------------------------
def _difficulty_factors(d):
    """Map difficulty d in [0,1] -> the privileged factors (vectorised).

    Weighted toward the factors RMA can actually identify from a 1-s history
    (added mass and COM shift), with only a mild friction component, so the
    no_adapt <= rma <= expert ordering is exposed rather than washed out by the
    weakly-identifiable friction/motor extremes.
    """
    return dict(
        payload=(d * 9.0).astype(np.float32),                   # 0   -> 9 kg
        com_xy=np.stack([d * 0.15, np.zeros_like(d)], 1).astype(np.float32),  # fwd
        friction=(1.0 + d * (0.7 - 1.0)).astype(np.float32),    # 1.0 -> 0.7 (mild)
        motor=np.ones_like(d, np.float32),                      # nominal
    )


def fig_difficulty_sweep(eng, out, N, T, nbins=8, modes=MODES, seed=0, refresh=False):
    cache = core.cache_path("difficulty")
    if not refresh and os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)["data"].item()
    else:
        d = np.linspace(0.0, 1.0, N).astype(np.float32)
        f = _difficulty_factors(d)
        cmd = core.const_commands(core.sample_commands(N, seed=seed), T)
        data = {}
        for m in modes:
            with core.timer(f"difficulty[{m}]"):
                r = eng.rollout(m, T, f["friction"], f["payload"], f["com_xy"],
                                f["motor"], cmd, seed=seed)
            data[m] = dict(d=d, surv=r.survival().astype(float), lin=r.ep_lin_err())
        np.savez(cache, data=data)

    edges = np.linspace(0, 1, nbins + 1)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.4))
    for m in modes:
        c, s, _ = _binned(data[m]["d"], data[m]["surv"], edges, np.mean)
        _, l, _ = _binned(data[m]["d"], data[m]["lin"], edges, np.median)
        a0.plot(c, s * 100, "-o", color=MODE_COLOR[m], lw=2, label=MODE_LABEL_SHORT[m])
        a1.plot(c, l, "-o", color=MODE_COLOR[m], lw=2)
    a0.set_ylabel("survival (%)"); a0.set_ylim(-3, 103); a0.legend(fontsize=9)
    a1.set_ylabel("lin-vel track err (m/s)")
    for ax in (a0, a1):
        ax.set_xlabel("compound difficulty  d"); ax.grid(alpha=0.3)
    fig.suptitle("Compound out-of-distribution difficulty "
                 "(added mass + COM shift, mild friction): the bounds separate",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Control-level adaptation: how the policy's torque / smoothness change with the
# environment (e.g. slippery vs nominal vs loaded). Shows that the adaptive modes
# modulate *control effort*, not just survive.
# ---------------------------------------------------------------------------
EFFORT_CONDS = [
    ("nominal",   dict(friction=1.0, payload=0.0, motor=1.0)),
    ("slippery",  dict(friction=0.3, payload=0.0, motor=1.0)),
    ("ice",       dict(friction=0.15, payload=0.0, motor=1.0)),
    ("+6 kg",     dict(friction=1.0, payload=6.0, motor=1.0)),
    ("weak 0.7x", dict(friction=1.0, payload=0.0, motor=0.7)),
]


def fig_control_effort(eng, out, N, T, modes=MODES, seed=0, refresh=False):
    cache = core.cache_path("control_effort")
    if not refresh and os.path.exists(cache):
        rows = np.load(cache, allow_pickle=True)["rows"].item()
    else:
        cmd = core.const_commands(core.sample_commands(N, seed=seed), T)
        rows = {}
        for name, p in EFFORT_CONDS:
            fr = np.full(N, p["friction"], np.float32)
            pl = np.full(N, p["payload"], np.float32)
            mo = np.full(N, p["motor"], np.float32)
            com = np.zeros((N, 2), np.float32)
            for m in modes:
                with core.timer(f"effort[{name},{m}]"):
                    r = eng.rollout(m, T, fr, pl, com, mo, cmd, seed=seed)
                a = r.survival()
                rows[(name, m)] = dict(
                    torque=float(np.median(r.torque_sq()[a])) if a.any() else np.nan,
                    smooth=float(np.median(r.smoothness()[a])) if a.any() else np.nan,
                    surv=a.mean() * 100)
        np.savez(cache, rows=rows)

    names = [n for n, _ in EFFORT_CONDS]
    x = np.arange(len(names)); w = 0.26
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    panels = [("torque", "control effort  ⟨‖τ‖²⟩"),
              ("smooth", "roughness  ⟨‖Δτ‖²⟩"),
              ("surv", "survival (%)")]
    for ax, (key, ylab) in zip(axes, panels):
        for i, m in enumerate(modes):
            ax.bar(x + (i - 1) * w, [rows[(n, m)][key] for n in names], w,
                   color=MODE_COLOR[m], label=MODE_LABEL_SHORT[m] if key == "torque" else None)
        ax.set_ylabel(ylab); ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right"); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle("Control-level adaptation: torque / smoothness modulate with the "
                 "terrain and load", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CONCLUSIVE adaptation-edge test, on the HEIGHTFIELD, with many runs + 95% CIs.
# Added trunk load is the regime where the fixed z=mu(0) of no_adapt is most wrong
# and phi's inference helps most -- so this is where no_adapt < rma < expert is
# cleanest. Each condition is a group of `n_per` envs in one batched rollout.
# ---------------------------------------------------------------------------
LOAD_CONDS = [("nominal", 0.0, 0.0), ("+4 kg", 4.0, 0.0), ("+6 kg", 6.0, 0.0),
              ("+8 kg", 8.0, 0.0), ("front-heavy\n(+5 kg, COM)", 5.0, 0.12)]


def _ci95(p, n):
    """95% normal-approx binomial confidence half-width for proportion p over n."""
    return 1.96 * np.sqrt(np.maximum(p * (1 - p), 1e-9) / max(n, 1))


def fig_load_bars(eng, out, n_per, T, modes=MODES, seed=0, z_scale=0.10, refresh=False):
    cache = core.cache_path("load_bars")
    if not refresh and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        rows = d["rows"].item(); n_per = int(d["n_per"])
    else:
        nC = len(LOAD_CONDS)
        N = n_per * nC
        payload = np.concatenate([np.full(n_per, p) for _, p, _ in LOAD_CONDS]).astype(np.float32)
        com_xy = np.concatenate([np.tile([c, 0.0], (n_per, 1)) for _, _, c in LOAD_CONDS]).astype(np.float32)
        fr = np.ones(N, np.float32); mo = np.ones(N, np.float32)
        heng = core.Engine(eng.ckpt1, eng.ckpt2, terrain="hfield", z_scale=z_scale)
        cmd = core.const_commands(core.sample_commands(N, seed=seed), T)
        rows = {}
        for m in modes:
            with core.timer(f"load_bars[{m}] N={N}"):
                r = heng.rollout(m, T, fr, payload, com_xy, mo, cmd, seed=seed)
            surv = r.survival(); tq = r.torque_sq()
            for g, (name, _, _) in enumerate(LOAD_CONDS):
                sl = slice(g * n_per, (g + 1) * n_per)
                p = surv[sl].mean()
                rows[(name, m)] = dict(surv=p * 100, ci=_ci95(p, n_per) * 100,
                                       torque=float(np.median(tq[sl][surv[sl]]))
                                       if surv[sl].any() else np.nan)
        np.savez(cache, rows=rows, n_per=n_per)

    names = [n for n, _, _ in LOAD_CONDS]
    x = np.arange(len(names)); w = 0.26
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, m in enumerate(modes):
        off = (i - 1) * w
        a0.bar(x + off, [rows[(n, m)]["surv"] for n in names], w, color=MODE_COLOR[m],
               yerr=[rows[(n, m)]["ci"] for n in names], capsize=3,
               error_kw=dict(lw=1, ecolor="0.3"), label=MODE_LABEL_SHORT[m])
        a1.bar(x + off, [rows[(n, m)]["torque"] for n in names], w, color=MODE_COLOR[m])
    a0.set_ylabel("survival (%)  ±95% CI"); a0.set_ylim(0, 105); a0.legend(fontsize=9)
    a1.set_ylabel("median control effort  ⟨‖τ‖²⟩")
    for ax in (a0, a1):
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Adaptation edge under load — MJX heightfield, {n_per} episodes / "
                 f"condition (95% CIs)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def fig_difficulty_hfield(eng, out, N, T, nbins=8, modes=MODES, seed=0,
                          z_scale=0.10, refresh=False):
    """Compound difficulty (mass+COM+mild friction) ON the training heightfield."""
    cache = core.cache_path("difficulty_hfield")
    if not refresh and os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)["data"].item()
    else:
        d = np.linspace(0.0, 1.0, N).astype(np.float32)
        f = _difficulty_factors(d)
        heng = core.Engine(eng.ckpt1, eng.ckpt2, terrain="hfield", z_scale=z_scale)
        cmd = core.const_commands(core.sample_commands(N, seed=seed), T)
        data = {}
        for m in modes:
            with core.timer(f"difficulty_hfield[{m}] N={N}"):
                r = heng.rollout(m, T, f["friction"], f["payload"], f["com_xy"],
                                 f["motor"], cmd, seed=seed)
            data[m] = dict(d=d, surv=r.survival().astype(float), lin=r.ep_lin_err())
        np.savez(cache, data=data)

    edges = np.linspace(0, 1, nbins + 1)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.4))
    for m in modes:
        c, s, cnt = _binned(data[m]["d"], data[m]["surv"], edges, np.mean)
        _, l, _ = _binned(data[m]["d"], data[m]["lin"], edges, np.median)
        ci = _ci95(s, cnt.mean()) * 100
        a0.errorbar(c, s * 100, yerr=ci, fmt="-o", color=MODE_COLOR[m], lw=2,
                    capsize=3, label=MODE_LABEL_SHORT[m])
        a1.plot(c, l, "-o", color=MODE_COLOR[m], lw=2)
    a0.set_ylabel("survival (%)  ±95% CI"); a0.set_ylim(-3, 103); a0.legend(fontsize=9)
    a1.set_ylabel("lin-vel track err (m/s)")
    for ax in (a0, a1):
        ax.set_xlabel("compound difficulty  d  (mass+COM+friction)"); ax.grid(alpha=0.3)
    fig.suptitle("Compound difficulty on the heightfield: the bounds separate "
                 "(±95% CI)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
