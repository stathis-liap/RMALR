"""Configuration for the RMA implementation.

Task: Project 3 - goal-conditioned Go2 velocity tracking (gym-quadruped). RMA is
trained in MJX, then a ``Controller`` runs the trained networks inside the
benchmark for grading (see ``rma/controller.py``, ``rma/eval_gym.py``).

Train<->deploy consistency hinges on three choices:
* The policy state ``x_t`` is built only from observations the proprioceptive
  benchmark exposes, in its joint order (``qpos[7:]`` = FL, FR, RL, RR).
* The velocity command enters ``x_t`` as the base-frame tracking errors the
  benchmark provides (goal conditioning).
* Training reproduces the Project-3 reward, so MJX optimises the graded objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Environment / robot
# ---------------------------------------------------------------------------
@dataclass
class EnvConfig:
    # gym-quadruped runs at 500 Hz; we match it and decimate by 10 -> 50 Hz
    # control (a common Go2 RL rate).
    physics_dt: float = 0.002
    control_decimation: int = 10
    episode_length: int = 1000         # control steps per episode

    # Fall thresholds. Base height is measured above the *local* ground, so these
    # hold on both flat and uneven terrain. Standing is ~0.27 m; 0.24 catches the
    # crouch-until-the-thighs-scrape failure (foot-only MJX collisions let the
    # trunk pass through the floor, so a height proxy stands in for body contact).
    min_base_height: float = 0.24      # m above local terrain
    upright_z_thresh: float = -0.5     # world-down in base z (-1 upright, 0 on side)

    # Reset-state noise, mirroring gym-quadruped's reset (which the grader also
    # applies): per-joint pose/vel noise, base roll/pitch tilt, and a small drop-in.
    reset_joint_pos_noise: float = 0.30   # rad
    reset_joint_vel_noise: float = 0.50   # rad/s
    reset_rp_noise: float = 0.15          # rad, roll & pitch
    reset_drop_height: float = 0.03       # m, spawn clearance above the ground
    reset_xy_range: float = 4.0           # m, random spawn xy (terrain variety)

    # Goal-conditioned velocity command g_t = [vx, vy, yaw_rate], base frame. Full
    # omnidirectional range matching the grader (a speed in [0,1] m/s with uniform
    # random heading, plus yaw in [-1,1]); resampled occasionally within an episode
    # so the policy learns command transitions.
    cmd_vx_range: Tuple[float, float] = (-1.0, 1.0)   # m/s
    cmd_vy_range: Tuple[float, float] = (-1.0, 1.0)   # m/s
    cmd_wz_range: Tuple[float, float] = (-1.0, 1.0)   # rad/s
    cmd_resample_prob: float = 0.005                  # per control step

    # Terrain. "hfield" trains on a procedural fractal heightfield (1 shared geom,
    # the MJX-efficient choice); per-env variety comes from the random spawn xy.
    # "scene" uses the flat plane the grader evaluates on. Either way collisions
    # are foot-only (MJX can't body<->hfield), and terrain stays gentle since the
    # benchmark is flat -- the heightfield only adds robustness (as in the paper).
    terrain: str = "hfield"            # {"hfield", "scene" (flat)}
    hfield_radius: float = 20.0        # m, half-extent (large enough to walk on)
    hfield_grid: int = 256             # cells per side
    fractal_octaves: int = 2
    fractal_lacunarity: float = 2.0
    fractal_gain: float = 0.25
    fractal_base_freq: int = 16        # coarsest fBm frequency (features/width)
    fractal_z_scale: float = 0.10      # m, peak elevation

    # Domain randomization (privileged e_t). kp floor is 25: below ~25 the robot
    # sags too far under gravity to stand, dooming those envs at birth.
    friction_range: Tuple[float, float] = (0.30, 2.0)
    kp_range: Tuple[float, float] = (25.0, 40.0)
    kd_range: Tuple[float, float] = (0.4, 1.0)
    payload_range: Tuple[float, float] = (0.0, 5.0)     # kg added to trunk
    com_range: Tuple[float, float] = (-0.10, 0.10)      # m trunk COM xy shift
    motor_strength_range: Tuple[float, float] = (0.85, 1.15)
    resample_prob: float = 0.004        # within-episode DR resample, per step

    # External pushes: occasional lateral base-velocity impulse so the policy
    # learns to recover its footing instead of relying on the pristine sim.
    push_prob: float = 0.01             # per control step
    push_lin_vel: float = 0.5           # m/s

    action_scale: float = 0.30          # action is a residual on the nominal pose
    history_len: int = 50               # k timesteps fed to the adaptation module


# ---------------------------------------------------------------------------
# Reward (Project-3 _compute_reward, reproduced for training)
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    # The graded reward's sigma=0.05 is so peaked it gives no gradient until the
    # robot is already within ~0.1 m/s of the command. eval_gym keeps 0.05 (the
    # true metric); training uses gentler sigmas for a usable gradient. These are
    # tightened (0.35->0.25) and the tracking weight raised to prioritise velocity
    # accuracy (target ~95%), while staying loose enough not to kill early motion.
    sigma_lin_vel: float = 0.05        # grading reference (eval_gym)
    sigma_ang_vel: float = 0.05
    train_sigma_lin_vel: float = 0.25  # tighter -> demands more precision
    train_sigma_ang_vel: float = 0.30
    w_tracking_lin: float = 3.0        # tracking dominates the other terms
    w_tracking_yaw: float = 1.5
    w_upright: float = 0.5
    # Body-motion penalties, gentler than the grader's 0.2/0.1: penalizing a real
    # gait's natural bob/weight-shift drives the policy into a rigid micro-shuffle.
    w_z_vel: float = 0.05
    w_roll_pitch_ang: float = 0.05
    w_torque: float = 1e-4
    w_action_rate: float = 1e-4        # on 50 Hz torque deltas (see env)
    w_termination: float = 2.0         # one-shot penalty on a fall

    # Gait shaping -> a deliberate, foot-lifting, normal-width walk instead of the
    # tucked-leg shuffle pure tracking discovers.
    w_feet_air_time: float = 1.0       # reward footfalls near air_time_target
    air_time_target: float = 0.40      # s, desired swing duration
    foot_contact_height: float = 0.04  # m, foot-center z below this = stance
    w_foot_clearance: float = 1.0      # penalize a swing foot that drags
    foot_clearance_target: float = 0.072  # m, swing-foot height (lowered ~20%)
    w_hip_deviation: float = 0.5       # keep hips near nominal -> normal width
    w_foot_slip: float = 1.0           # penalize horizontal speed of stance feet

    # Penalties ramp from k0 to 1 via k_{t+1}=k_t^decay (avoids an early all-
    # negative reward that collapses to standing still).
    penalty_curriculum_k0: float = 0.1
    penalty_curriculum_decay: float = 0.997


# ---------------------------------------------------------------------------
# Network architecture (paper Sec IV-B)
# ---------------------------------------------------------------------------
@dataclass
class NetConfig:
    # x_t = gravity_base(3) + base_ang_vel(3) + qpos_js(12) + qvel_js(12)
    #       + base_lin_vel_err(3) + base_ang_vel_err(3) = 36
    state_dim: int = 36
    action_dim: int = 12
    # e_t = payload(1) + com_xy(2) + motor_strength(12) + friction(1)
    #       + terrain_height(1) = 17
    env_factor_dim: int = 17
    latent_dim: int = 8                # extrinsics z

    policy_hidden: Tuple[int, ...] = (128, 128, 128)
    encoder_hidden: Tuple[int, ...] = (256, 128)
    value_hidden: Tuple[int, ...] = (256, 256, 256)

    # Adaptation module: 2-layer MLP embed -> 32, then 3-layer 1-D CNN.
    adapt_embed_hidden: Tuple[int, ...] = (32,)
    adapt_embed_dim: int = 32
    adapt_conv: Tuple[Tuple[int, int, int, int], ...] = (
        (32, 32, 8, 4),
        (32, 32, 5, 1),
        (32, 32, 5, 1),
    )
    # Exploration std, clamped to [min, max]. The floor keeps stepping-scale
    # noise alive; the ceiling stops the entropy bonus inflating std until the
    # action noise destabilizes the gait.
    log_std_init: float = -0.5          # std ~0.61
    min_log_std: float = -1.40          # std floor ~0.25
    max_log_std: float = -0.357         # std ceiling ~0.70


# ---------------------------------------------------------------------------
# PPO (Phase 1)
# ---------------------------------------------------------------------------
@dataclass
class PPOConfig:
    num_envs: int = 4096
    unroll_length: int = 24
    num_iterations: int = 15000
    num_minibatches: int = 4
    num_epochs: int = 4
    learning_rate: float = 5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    # Entropy bonus keeps std off its floor (without it the policy collapses to a
    # stand-still that tracks nothing). 0.004 explores early but lets the policy
    # gradient pull std down for precision later.
    entropy_coef: float = 0.004
    max_grad_norm: float = 1.0
    seed: int = 0
    save_every: int = 250


# ---------------------------------------------------------------------------
# Adaptation module training (Phase 2)
# ---------------------------------------------------------------------------
@dataclass
class AdaptConfig:
    num_envs: int = 4096
    unroll_length: int = 50            # >= history_len for valid windows
    num_iterations: int = 2000
    num_minibatches: int = 4
    learning_rate: float = 5e-4
    seed: int = 1
    save_every: int = 50


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    net: NetConfig = field(default_factory=NetConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)

    # "auto" -> the Go2 MJCF bundled in the gym-quadruped pip package, so MJX
    # training and the grader use the exact same model. Override with a path to a
    # custom scene if desired.
    model_path: str = "auto"
    checkpoint_dir: str = "checkpoints"
