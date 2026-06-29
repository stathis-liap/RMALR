"""Evaluate the RMA Controller inside the gym-quadruped Project-3 benchmark.

This mirrors the environment setup from the project brief (proprioceptive
observation variant, IMU sensor, Project-3 reward) and drives it with
``rma.controller.Controller``. It reports the metrics the grading emphasises:
velocity-tracking accuracy, survival, and episode reward.

Usage:
  python -m rma.eval_gym --phase1 checkpoints/phase1_final.pkl \
                         --phase2 checkpoints/phase2_final.pkl \
                         --episodes 20 --scene flat
"""
from __future__ import annotations

import argparse

import numpy as np
import mujoco

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.sensors.imu import IMU

from .controller import Controller


# --- Project-3 IMU init patch: step the sim once before the IMU reads data ---
_original_imu_init = IMU.__init__


def _patched_imu_init(self, mj_model, mj_data, *args, **kwargs):
    mujoco.mj_forward(mj_model, mj_data)
    _original_imu_init(self, mj_model, mj_data, *args, **kwargs)
    self.step()


IMU.__init__ = _patched_imu_init


# --- Project-3 reward (used only for reporting; controller never sees it) -----
def _compute_reward(self):
    lin_vel_err_B = self.base_lin_vel_err(frame="base")
    ang_vel_err_B = self.base_ang_vel_err(frame="base")

    sigma_lin_vel = 0.05
    sigma_ang_vel = 0.05
    tracking_lin_vel = np.exp(-np.sum(lin_vel_err_B[:2] ** 2) / (2 * sigma_lin_vel ** 2))
    tracking_yaw_rate = np.exp(-(ang_vel_err_B[2] ** 2) / (2 * sigma_ang_vel ** 2))

    gravity_B = self.gravity_vector
    upright_penalty = np.sum(gravity_B[:2] ** 2)

    base_lin_vel_B = self.base_lin_vel(frame="base")
    base_ang_vel_B = self.base_ang_vel(frame="base")
    z_vel_penalty = base_lin_vel_B[2] ** 2
    roll_pitch_ang_vel_penalty = np.sum(base_ang_vel_B[:2] ** 2)

    tau = self.torque_ctrl_setpoint
    torque_penalty = np.sum(tau ** 2)

    current_action = self.mjData.ctrl.copy()
    if not hasattr(self, "_last_action_for_reward"):
        self._last_action_for_reward = np.zeros_like(current_action)
    action_rate_penalty = np.sum((current_action - self._last_action_for_reward) ** 2)
    self._last_action_for_reward = current_action

    reward = (
        2.0 * tracking_lin_vel
        + 1.0 * tracking_yaw_rate
        - 0.5 * upright_penalty
        - 0.2 * z_vel_penalty
        - 0.1 * roll_pitch_ang_vel_penalty
        - 1e-4 * torque_penalty
        - 0.01 * action_rate_penalty
    )
    return float(reward)


QuadrupedEnv._compute_reward = _compute_reward


PROPRIOCEPTIVE_OBS = (
    "gravity_vector:base",
    "imu_acc",
    "imu_gyro",
    "qpos_js",
    "qvel_js",
    "base_lin_vel_err:base",
    "base_ang_vel_err:base",
)


def make_env(scene: str, friction=(0.4, 1.2), payload: float = 0.0):
    """Build the eval env.

    friction: ground friction range (lo, hi) sampled per episode. Pass (f, f) to
        pin a fixed friction for a domain-shift sweep.
    payload: kg added to the trunk body (0 = nominal). Tests load adaptation --
        the regime where RMA phi should beat the fixed mu(e_nominal) of no_adapt.
    """
    imu_kwargs = {"accel_name": "imu_acc", "gyro_name": "imu_gyro",
                  "imu_site_name": "imu"}
    env = QuadrupedEnv(
        robot="go2",
        scene=scene,
        base_vel_command_type="random_rotate_reset",
        ref_base_lin_vel=(0.0, 1.0),
        ref_base_ang_vel=(-1.0, 1.0),
        ground_friction_coeff=tuple(friction),
        state_obs_names=PROPRIOCEPTIVE_OBS,
        sensors=(IMU,),
        sensors_kwargs=(imu_kwargs,),
    )
    if payload:
        for name in ("base", "base_link", "trunk"):
            bid = mujoco.mj_name2id(env.mjModel, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                env.mjModel.body_mass[bid] += float(payload)
                break
    return env


def run(args):
    friction = ((args.friction, args.friction) if args.friction is not None
                else (0.4, 1.2))
    env = make_env(args.scene, friction=friction, payload=args.payload)
    ctrl = Controller(
        phase1_ckpt=args.phase1,
        phase2_ckpt=(args.phase2 if args.mode == "rma" else None),
        kp=args.kp, kd=args.kd, mode=args.mode,
    )

    ep_rewards, ep_lin_err, ep_yaw_err, ep_len, survived = [], [], [], [], []
    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed_base + ep)
        ctrl.reset(seed=args.seed_base + ep)
        terminated = truncated = False
        total_r, lin_errs, yaw_errs, steps = 0.0, [], [], 0
        last_info = {}
        while not (terminated or truncated) and steps < args.max_steps:
            action, _ = ctrl.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            last_info = info
            total_r += reward
            lin_errs.append(np.linalg.norm(obs["base_lin_vel_err:base"][:2]))
            yaw_errs.append(abs(obs["base_ang_vel_err:base"][2]))
            steps += 1
        if terminated:
            # diagnose *why* it fell: which bodies hit the ground + base state
            bad = list(last_info.get("invalid_contacts", {}).keys())
            try:
                bh = float(env.base_pos[2]); gz = float(env.gravity_vector[2])
            except Exception:
                bh, gz = float("nan"), float("nan")
            print(f"     -> term: base_h={bh:.3f} grav_z={gz:.2f} "
                  f"contacts={bad if bad else 'none(out-of-bounds?)'}")
        ep_rewards.append(total_r)
        ep_lin_err.append(np.mean(lin_errs))
        ep_yaw_err.append(np.mean(yaw_errs))
        ep_len.append(steps)
        survived.append(not terminated)
        print(f"[ep {ep:02d}] reward={total_r:8.1f} steps={steps:4d} "
              f"lin_err={np.mean(lin_errs):.3f} yaw_err={np.mean(yaw_errs):.3f} "
              f"{'survived' if not terminated else 'FELL'}")
    env.close()

    fric_str = (f"{args.friction}" if args.friction is not None else "0.4-1.2")
    print("\n=== gym-quadruped Go2 velocity-tracking "
          f"(mode={args.mode}, scene={args.scene}, n={args.episodes}, "
          f"friction={fric_str}, payload={args.payload}kg, "
          f"seed_base={args.seed_base}) ===")
    print(f"survival rate     : {100.0 * np.mean(survived):.1f}%")
    print(f"episode reward    : {np.mean(ep_rewards):.1f} +/- {np.std(ep_rewards):.1f}")
    print(f"episode length    : {np.mean(ep_len):.0f}")
    print(f"lin-vel track err : {np.mean(ep_lin_err):.4f} m/s")
    print(f"yaw-rate track err: {np.mean(ep_yaw_err):.4f} rad/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", type=str, default="checkpoints/phase1_final.pkl")
    ap.add_argument("--phase2", type=str, default="checkpoints/phase2_final.pkl")
    ap.add_argument("--mode", type=str, default="rma", choices=["rma", "no_adapt"])
    ap.add_argument("--scene", type=str, default="flat")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--kp", type=float, default=30.0)
    ap.add_argument("--kd", type=float, default=0.7)
    ap.add_argument("--seed-base", type=int, default=0,
                    help="base seed; episode e uses seed_base+e (same e -> same "
                         "command+friction across modes, for paired comparison)")
    ap.add_argument("--friction", type=float, default=None,
                    help="pin a fixed ground friction (overrides the 0.4-1.2 "
                         "range) for a domain-shift sweep, e.g. 0.3 or 1.5")
    ap.add_argument("--payload", type=float, default=0.0,
                    help="kg added to the trunk (tests load adaptation)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
