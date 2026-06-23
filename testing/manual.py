"""Manual teleop: drive the trained policy with the keyboard in a live viewer.

Uses gym-quadruped's own ``render()`` (so the robot shows and the camera follows)
and installs our command keys on its viewer. The typed velocity command is injected
by overwriting the two tracking-error observations the Controller reads
(``base_lin_vel_err`` / ``base_ang_vel_err``) with ``command - actual_velocity``,
so the policy chases whatever you type.

Movement is on I/J/K/L because gym-quadruped's viewer already binds the arrow
keys (and space / ctrl) for its own use -- these keys don't collide:

    I / K        : forward / backward
    J / L        : strafe left / right
    W / S        : increase / decrease speed
    A / D        : turn left / right (yaw)
    X            : stop (zero the command)
    R            : reset the episode
    close window : quit

Run with the GLFW backend so the window opens:
    MUJOCO_GL=glfw python -m testing.manual --scene flat \
        --phase1 checkpoints/phase1_final.pkl --phase2 checkpoints/phase2_final.pkl
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from rma.controller import Controller
from .runner import build_env
from .scenarios import Scenario

SPEED_STEP, YAW_STEP = 0.1, 0.2
SPEED_MAX, YAW_MAX = 1.2, 1.0


class Command:
    """Keyboard-driven velocity command: vx, vy (m/s) and wz (rad/s)."""

    def __init__(self, speed=0.5):
        self.vx = self.vy = self.wz = 0.0
        self.speed = speed
        self.reset_flag = False

    def on_key(self, key):
        if key == ord("I"):
            self.vx, self.vy = self.speed, 0.0
        elif key == ord("K"):
            self.vx, self.vy = -self.speed, 0.0
        elif key == ord("J"):
            self.vx, self.vy = 0.0, self.speed
        elif key == ord("L"):
            self.vx, self.vy = 0.0, -self.speed
        elif key == ord("W"):
            self.speed = min(SPEED_MAX, self.speed + SPEED_STEP); self._rescale()
        elif key == ord("S"):
            self.speed = max(0.0, self.speed - SPEED_STEP); self._rescale()
        elif key == ord("A"):
            self.wz = min(YAW_MAX, self.wz + YAW_STEP)
        elif key == ord("D"):
            self.wz = max(-YAW_MAX, self.wz - YAW_STEP)
        elif key == ord("X"):
            self.vx = self.vy = self.wz = 0.0
        elif key == ord("R"):
            self.reset_flag = True
        else:
            return                       # ignore keys we don't handle
        self._print()

    def _rescale(self):
        if self.vx:
            self.vx = np.sign(self.vx) * self.speed
        if self.vy:
            self.vy = np.sign(self.vy) * self.speed

    def _print(self):
        print(f"\rcmd  vx={self.vx:+.2f}  vy={self.vy:+.2f}  yaw={self.wz:+.2f}"
              f"  speed={self.speed:.2f}   ", end="", flush=True)


def _inject_command(obs, env, cmd):
    """Overwrite the error obs with (command - actual base velocity), base frame."""
    v = np.asarray(env.base_lin_vel(frame="base"))
    w = np.asarray(env.base_ang_vel(frame="base"))
    obs["base_lin_vel_err:base"] = np.array(
        [cmd.vx - v[0], cmd.vy - v[1], 0.0 - v[2]], dtype=np.float32)
    obs["base_ang_vel_err:base"] = np.array(
        [0.0 - w[0], 0.0 - w[1], cmd.wz - w[2]], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1", default="checkpoints/phase1_final.pkl")
    ap.add_argument("--phase2", default="checkpoints/phase2_final.pkl")
    ap.add_argument("--mode", default="rma", choices=["rma", "no_adapt"])
    ap.add_argument("--scene", default="flat", help="flat | perlin | random_boxes")
    ap.add_argument("--kp", type=float, default=30.0)
    ap.add_argument("--kd", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = build_env(Scenario("manual", scene=args.scene))
    ctrl = Controller(phase1_ckpt=args.phase1,
                      phase2_ckpt=(args.phase2 if args.mode == "rma" else None),
                      kp=args.kp, kd=args.kd, mode=args.mode)
    obs = env.reset(seed=args.seed)
    ctrl.reset(seed=args.seed)
    cmd = Command()

    # render() wires the viewer's keys to ``env._key_callback``; replace it with
    # ours. (gym-quadruped's own handler only works in its keyboard-command mode
    # and errors otherwise, so we don't chain it; it owned the arrow keys anyway,
    # which our handler simply ignores.)
    env._key_callback = cmd.on_key

    print(__doc__.split("Run with")[0])      # show the keymap
    step, t0 = 0, time.time()
    while True:
        if cmd.reset_flag:
            obs = env.reset(); ctrl.reset(); cmd.reset_flag = False
            step, t0 = 0, time.time()
        _inject_command(obs, env, cmd)
        action, _ = ctrl.act(obs)
        obs, _, term, trunc, _ = env.step(action)
        step += 1
        if term or trunc:                     # fell or timed out -> respawn
            obs = env.reset(); ctrl.reset()
            step, t0 = 0, time.time()
        env.render()                          # opens the window on first call
        if env.viewer is not None and not env.viewer.is_running():
            break
        lag = step * 0.002 - (time.time() - t0)   # pace to ~real time
        if lag > 0:
            time.sleep(min(lag, 0.02))
    env.close()


if __name__ == "__main__":
    main()
