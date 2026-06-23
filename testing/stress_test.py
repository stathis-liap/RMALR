"""Stress-test the trained RMA policy across out-of-distribution scenarios.

Examples:
  # one preset
  python -m testing.stress_test --scenario slippery --episodes 20 \
      --phase1 checkpoints/phase1_final.pkl --phase2 checkpoints/phase2_final.pkl

  # the whole suite, printed as a table
  python -m testing.stress_test --scenario all --episodes 20

  # watch it (needs a display): MUJOCO_GL=glfw
  python -m testing.stress_test --scenario gauntlet --episodes 3 --render

  # lower bound (no_adapt) vs RMA vs upper bound (expert) on every scenario
  python -m testing.stress_test --scenario all --mode all --episodes 20

  # ad-hoc: override stressors on top of (or instead of) a preset
  python -m testing.stress_test --scene perlin --friction 0.25 --push-vel 0.6 --push-interval 100

  python -m testing.stress_test --list      # list presets
"""
from __future__ import annotations

import argparse
import dataclasses

from rma.controller import Controller
from .scenarios import SCENARIOS, Scenario
from .runner import run_scenario, summarize


def _apply_overrides(sc: Scenario, args) -> Scenario:
    """Return a copy of `sc` with any explicitly-passed CLI stressors overridden."""
    ov = {}
    if args.scene is not None:
        ov["scene"] = args.scene
    if args.friction is not None:
        ov["friction"] = (args.friction, args.friction)
    if args.payload is not None:
        ov["payload"] = args.payload
    if args.com_x is not None:
        ov["com_shift"] = (args.com_x, 0.0)
    if args.motor_scale is not None:
        ov["motor_scale"] = args.motor_scale
    if args.push_vel is not None:
        ov["push_vel"] = args.push_vel
    if args.push_interval is not None:
        ov["push_interval"] = args.push_interval
    return dataclasses.replace(sc, **ov) if ov else sc


def _build_controller(mode, args):
    return Controller(
        phase1_ckpt=args.phase1,
        phase2_ckpt=(args.phase2 if (mode == "rma" and args.phase2) else None),
        kp=args.kp, kd=args.kd, mode=mode,
    )


def _print_table(rows):
    print("\n" + "=" * 82)
    print(f"{'scenario':<18}{'mode':<11}{'survival':>10}{'steps':>9}"
          f"{'lin_err':>10}{'yaw_err':>10}")
    print("-" * 82)
    for r in rows:
        print(f"{r['scenario']:<18}{r['mode']:<11}{r['survival']:>9.0f}%"
              f"{r['steps']:>9.0f}{r['lin_err']:>10.3f}{r['yaw_err']:>10.3f}")
    print("=" * 82)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="nominal",
                    help="preset name (see --list) or 'all'")
    ap.add_argument("--list", action="store_true", help="list presets and exit")
    ap.add_argument("--phase1", default="checkpoints/phase1_final.pkl")
    ap.add_argument("--phase2", default="checkpoints/phase2_final.pkl",
                    help="phase-2 ckpt; pass '' for mode=no_adapt")
    ap.add_argument("--mode", default="rma",
                    choices=["rma", "no_adapt", "expert", "all"],
                    help="'all' = no_adapt (lower bound) + rma + expert (upper "
                         "bound), the three baselines side by side")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--kp", type=float, default=30.0)
    ap.add_argument("--kd", type=float, default=0.7)
    ap.add_argument("--render", action="store_true",
                    help="show the viewer (needs a display; use MUJOCO_GL=glfw)")
    ap.add_argument("--hfield", action="store_true",
                    help="run on the MJX training heightfield (faithful terrain) "
                         "instead of gym-quadruped; --episodes = parallel envs")
    ap.add_argument("--hfield-z", type=float, default=0.10,
                    help="heightfield height for --hfield (0.10 = training)")
    # ad-hoc stressor overrides
    ap.add_argument("--scene", default=None,
                    help="flat | perlin | random_boxes")
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--payload", type=float, default=None)
    ap.add_argument("--com-x", type=float, default=None)
    ap.add_argument("--motor-scale", type=float, default=None)
    ap.add_argument("--push-vel", type=float, default=None)
    ap.add_argument("--push-interval", type=int, default=None)
    args = ap.parse_args()

    if args.list:
        print("Available scenarios:")
        for name, sc in SCENARIOS.items():
            print(f"  {name:<18} {sc.description}")
        print("  all                run every scenario above")
        return

    if args.scenario != "all" and args.scenario not in SCENARIOS:
        ap.error(f"unknown scenario '{args.scenario}'. Use --list to see options.")

    # no_adapt (lower bound) -> rma -> expert (upper bound)
    modes = ["no_adapt", "rma", "expert"] if args.mode == "all" else [args.mode]
    if "rma" in modes and not args.phase2:
        ap.error("mode 'rma'/'all' needs a phase-2 checkpoint (--phase2).")
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    # --- MJX training-heightfield path: batched rollout, no Controller --------
    if args.hfield:
        from .hfield_stress import run_scenario_hfield
        rows = []
        for name in names:
            sc = _apply_overrides(SCENARIOS[name], args)
            for mode in modes:
                print(f"\n### {sc.name} [{mode}] on hfield: {sc.description}", flush=True)
                rows.append(run_scenario_hfield(
                    sc, args.phase1, args.phase2, mode, n_envs=args.episodes,
                    max_steps=args.max_steps, seed=args.seed_base, z_scale=args.hfield_z))
        _print_table(rows)
        return

    # --- gym-quadruped path (flat / perlin / boxes scenes) -------------------
    controllers = {m: _build_controller(m, args) for m in modes}
    rows = []
    for name in names:
        sc = _apply_overrides(SCENARIOS[name], args)
        for mode in modes:
            print(f"\n### {sc.name} [{mode}]: {sc.description}")
            results = run_scenario(sc, controllers[mode], args.episodes,
                                   args.max_steps, args.seed_base, render=args.render)
            rows.append(summarize(sc.name, mode, results))

    _print_table(rows)


if __name__ == "__main__":
    main()
