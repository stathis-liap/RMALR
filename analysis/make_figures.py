"""Generate the full RMA figure suite.

  # everything, quick (laptop / CPU, a few minutes):
  python -m analysis.make_figures --quick

  # full quality (run on the GPU training server):
  python -m analysis.make_figures --full

  # one figure, recompute its cache:
  python -m analysis.make_figures --only tracking_timeseries --refresh

Figures land in ``figures/`` (raw arrays cached in ``figures/cache/`` so a
re-style is instant). ``--quick`` shrinks batch sizes / horizons for a fast
local pass; ``--full`` is sized for the GPU server. See ``analysis/README.md``
for what each figure shows and how it maps to the paper and Project 3.
"""
from __future__ import annotations

import argparse
import os
import time

from . import core, tracking, robustness, adaptation

# (name -> (callable, kwargs-key)). Each builds one figure file.
FIGURES = {
    # Project-3 core: goal-conditioned velocity tracking
    "tracking_timeseries": lambda eng, s: tracking.fig_tracking_timeseries(
        eng, core.fig_path("01_tracking_timeseries.png"), **s["track_ts"]),
    "tracking_accuracy": lambda eng, s: tracking.fig_tracking_accuracy(
        eng, core.fig_path("02_tracking_accuracy.png"), **s["track_acc"]),
    "tracking_polar": lambda eng, s: tracking.fig_tracking_polar(
        eng, core.fig_path("03_tracking_polar.png"), **s["track_polar"]),
    # robustness / baselines
    "generalization": lambda eng, s: robustness.fig_generalization_sweeps(
        eng, core.fig_path("04_generalization_sweeps.png"), **s["gen"]),
    "stress_bars": lambda eng, s: robustness.fig_stress_bars(
        eng, core.fig_path("05_stress_scenarios.png"), **s["stress"]),
    "summary_table": lambda eng, s: robustness.summary_table(
        eng, core.fig_path("06_summary_table.png"), **s["summary"]),
    "hfield_sweep": lambda eng, s: robustness.fig_hfield_sweep(
        eng, core.fig_path("09_hfield_terrain_sweep.png"), **s["hfield"]),
    "sensor_noise": lambda eng, s: robustness.fig_sensor_noise(
        eng, core.fig_path("10_sensor_noise.png"), **s["noise"]),
    "difficulty": lambda eng, s: robustness.fig_difficulty_sweep(
        eng, core.fig_path("11_compound_difficulty.png"), **s["difficulty"]),
    "control_effort": lambda eng, s: robustness.fig_control_effort(
        eng, core.fig_path("12_control_effort.png"), **s["effort"]),
    "load_bars": lambda eng, s: robustness.fig_load_bars(
        eng, core.fig_path("13_load_bars_hfield.png"), **s["loadbars"]),
    "difficulty_hfield": lambda eng, s: robustness.fig_difficulty_hfield(
        eng, core.fig_path("14_difficulty_hfield.png"), **s["diff_hf"]),
    # adaptation module
    "embedding": lambda eng, s: adaptation.fig_latent_embedding(
        eng, core.fig_path("07_latent_embedding.png"), **s["embed"]),
    "adaptation": lambda eng, s: adaptation.fig_adaptation(
        eng, core.fig_path("08_adaptation.png"), **s["adapt"]),
}


def settings(quick, refresh):
    """Per-figure (N envs, T steps) sized for a quick local pass or full server run."""
    if quick:
        s = dict(
            track_ts=dict(N=48, T=300),
            track_acc=dict(N=28, T=200),
            track_polar=dict(N=28, T=200),
            gen=dict(N=120, T=260, nbins=6),
            stress=dict(N=48, T=320),
            summary=dict(N=160, T=320),
            hfield=dict(N=64, T=260),
            noise=dict(N=80, T=260),
            difficulty=dict(N=512, T=300, nbins=8),
            effort=dict(N=96, T=260),
            loadbars=dict(n_per=128, T=250),
            diff_hf=dict(N=400, T=250, nbins=8),
            embed=dict(N=140, T=240),
            adapt=dict(N=64, T=300),
        )
    else:
        s = dict(
            track_ts=dict(N=96, T=480),
            track_acc=dict(N=60, T=320),
            track_polar=dict(N=48, T=320),
            gen=dict(N=420, T=500, nbins=9),
            stress=dict(N=256, T=600),
            summary=dict(N=512, T=600),
            hfield=dict(N=256, T=500),
            noise=dict(N=256, T=500),
            difficulty=dict(N=1024, T=500, nbins=10),
            effort=dict(N=256, T=500),
            loadbars=dict(n_per=400, T=500),
            diff_hf=dict(N=1024, T=500, nbins=10),
            embed=dict(N=400, T=400),
            adapt=dict(N=160, T=500),
        )
    for v in s.values():
        v["refresh"] = refresh
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="small batches/horizons for a fast CPU pass")
    ap.add_argument("--full", action="store_true",
                    help="full quality (GPU server); default if neither given")
    ap.add_argument("--only", nargs="+", choices=list(FIGURES),
                    help="generate only these figures")
    ap.add_argument("--refresh", action="store_true",
                    help="recompute cached rollouts instead of reusing them")
    ap.add_argument("--phase1", default=core.CKPT1)
    ap.add_argument("--phase2", default=core.CKPT2)
    args = ap.parse_args()

    quick = args.quick and not args.full
    mode = "quick" if quick else "full"

    # Cached rollouts are sized per mode; if the last run used the other mode,
    # force a refresh so a --full run doesn't silently reuse quick-resolution data.
    os.makedirs(core.CACHE_DIR, exist_ok=True)
    marker = os.path.join(core.CACHE_DIR, ".mode")
    prev = open(marker).read().strip() if os.path.exists(marker) else None
    if prev and prev != mode and not args.refresh:
        print(f"(cache was '{prev}', now '{mode}' -> forcing --refresh)")
        args.refresh = True
    open(marker, "w").write(mode)

    s = settings(quick, args.refresh)
    names = args.only or list(FIGURES)

    print(f"=== RMA figure suite ({mode}) -> figures/ ===")
    eng = core.Engine(args.phase1, args.phase2)
    for name in names:
        t0 = time.time()
        print(f"\n[{name}]")
        FIGURES[name](eng, s)
        print(f"  done in {time.time() - t0:.0f}s")
    print("\nAll figures written to figures/. See analysis/README.md.")


if __name__ == "__main__":
    main()
