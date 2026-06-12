"""Watch a trained RMA checkpoint drive the Go2 in the gym-quadruped benchmark.

Runs the exact deployment path used for grading (gym-quadruped env + the
``rma.controller.Controller``) with the on-screen MuJoCo viewer, so what you
see is what the evaluator scores.

Get a checkpoint from the training server first, e.g.:

    scp <user>@<server>:~/rmalr/checkpoints/phase1_1000.pkl checkpoints/

Then run locally (needs a display; use the project venv, NOT docker):

    python tools/view_policy.py --phase1 checkpoints/phase1_1000.pkl
    python tools/view_policy.py --phase1 checkpoints/phase1_final.pkl \
                                --phase2 checkpoints/phase2_final.pkl --mode rma

Without --phase2 it runs mode='no_adapt' (base policy, z = mu(0)) which is the
right way to inspect a Phase-1-only checkpoint.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", type=str, default="checkpoints/phase1_final.pkl")
    ap.add_argument("--phase2", type=str, default=None,
                    help="phase-2 checkpoint; omit to run mode='no_adapt'")
    ap.add_argument("--mode", type=str, default=None,
                    choices=["rma", "no_adapt"],
                    help="default: 'rma' if --phase2 given, else 'no_adapt'")
    ap.add_argument("--scene", type=str, default="flat")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render-every", type=int, default=10,
                    help="render every N env steps (10 = 50 fps at 500 Hz sim)")
    args = ap.parse_args()

    mode = args.mode or ("rma" if args.phase2 else "no_adapt")

    # Imported here so --help works without the heavy deps.
    from rma.eval_gym import make_env
    from rma.controller import Controller

    env = make_env(args.scene)
    ctrl = Controller(phase1_ckpt=args.phase1, phase2_ckpt=args.phase2, mode=mode)

    print(f"mode={mode} scene={args.scene} -- close the viewer window to stop")
    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep)
        ctrl.reset(seed=args.seed + ep)
        terminated = truncated = False
        total_r, steps = 0.0, 0
        lin_errs = []
        t_wall = time.time()
        while not (terminated or truncated) and steps < args.max_steps:
            action, _ = ctrl.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            lin_errs.append(np.linalg.norm(obs["base_lin_vel_err:base"][:2]))
            if steps % args.render_every == 0:
                env.render()
                # hold to ~real time (sim_dt=0.002 * render_every)
                target = steps * 0.002
                lag = target - (time.time() - t_wall)
                if lag > 0:
                    time.sleep(min(lag, 0.05))
            steps += 1
        print(f"[ep {ep:02d}] steps={steps:5d} reward={total_r:8.1f} "
              f"lin_err={np.mean(lin_errs):.3f} "
              f"{'FELL' if terminated else 'survived'}")
    env.close()


if __name__ == "__main__":
    main()
