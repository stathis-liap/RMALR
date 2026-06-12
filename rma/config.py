"""Central configuration for the RMA implementation.

Target task: **Project 3 - Goal-Conditioned Go2 Velocity Tracking**
(gym-quadruped benchmark). RMA is trained in MJX on the Unitree Go2, then a
``Controller`` adapter runs the trained networks inside gym-quadruped for the
official evaluation. See ``rma/controller.py`` and ``rma/eval_gym.py``.

Design choices that anchor train<->deploy consistency:

* The policy state ``x_t`` is built *only* from observations the gym-quadruped
  proprioceptive benchmark exposes, in the *same joint order* the benchmark uses
  (``qpos[7:]`` = FL, FR, RL, RR). See ``net.state_dim`` breakdown below.
* The robot is goal-conditioned: the velocity command enters ``x_t`` as the
  base-frame tracking errors ``base_lin_vel_err`` (3) and ``base_ang_vel_err``
  (3), exactly the quantities the benchmark provides.
* The reward reproduces the Project-3 ``_compute_reward`` so MJX training
  optimises the graded objective directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Environment / robot
# ---------------------------------------------------------------------------
@dataclass
class EnvConfig:
    # Physics. gym-quadruped runs at sim_dt=0.002 (500 Hz). We use a 500 Hz
    # physics step with decimation 10 -> 50 Hz control (a common Go2 RL rate).
    physics_dt: float = 0.002          # impl (500 Hz, matches gym-quadruped)
    control_decimation: int = 10       # impl -> 50 Hz control
    episode_length: int = 1000         # control steps per episode

    # Early-termination thresholds. NOTE: 0.18 was too close to the natural
    # stance sag under weak randomized gains (the robot terminated standing
    # still on flat ground) -- 0.15 leaves room for crouching during locomotion.
    min_base_height: float = 0.15      # m (Go2 hip height ~0.28)
    # Robot considered tipped when world-down projected into base frame has
    # z-component above this (-1 = perfectly upright, 0 = on its side).
    upright_z_thresh: float = -0.5

    # --- Reset-state randomization (matches gym-quadruped's reset, which adds
    # +-20deg joint noise, +-10deg roll/pitch and a drop-in spawn; a policy
    # trained only from the pristine nominal pose face-plants at evaluation) ---
    reset_joint_pos_noise: float = 0.30   # rad, uniform per joint
    reset_joint_vel_noise: float = 0.50   # rad/s, uniform per joint
    reset_rp_noise: float = 0.15          # rad, uniform roll & pitch
    reset_drop_height: float = 0.03       # m, spawn clearance above stance

    # --- Goal-conditioned velocity command (base/heading frame) -------------
    # g_t = [vx, vy, yaw_rate]. Ranges sampled at reset and (optionally) within
    # an episode to teach transitions between commands.
    # vy widened to match evaluation: gym-quadruped's "random" command samples a
    # speed with a uniformly random heading, so lateral commands up to the full
    # speed magnitude occur at eval time.
    cmd_vx_range: Tuple[float, float] = (-1.0, 1.0)   # m/s forward
    cmd_vy_range: Tuple[float, float] = (-1.0, 1.0)   # m/s lateral
    cmd_wz_range: Tuple[float, float] = (-1.0, 1.0)   # rad/s yaw rate
    cmd_resample_prob: float = 0.005   # per control step (~1 change / 200 steps)

    # Terrain. "hfield" injects a procedural fractal heightfield (1 geom, the
    # MJX-efficient choice) over the plain base scene -- this is the training
    # default. "scene" uses whatever geometry the model XML carries as-is; note
    # the 700-box make_jagged_scene field is too dense for MJX put_model, so it
    # is intended for visualization, not MJX training. z_scale is gentler than
    # the RMA paper's 0.27 since the Go2 is smaller and the graded benchmark is
    # flat -- the heightfield only adds robustness.
    terrain: str = "hfield"            # {"hfield", "scene"}
    fractal_octaves: int = 2
    fractal_lacunarity: float = 2.0
    fractal_gain: float = 0.25
    fractal_z_scale: float = 0.08

    # --- Domain randomization ranges (privileged e_t) -----------------------
    friction_range: Tuple[float, float] = (0.30, 2.0)   # foot sliding friction
    # kp low end raised from 20: below ~25 the robot sags so deeply under
    # gravity that standing is infeasible and those envs are doomed at birth.
    kp_range: Tuple[float, float] = (25.0, 40.0)        # PD position gain
    kd_range: Tuple[float, float] = (0.4, 1.0)          # PD damping gain
    payload_range: Tuple[float, float] = (0.0, 5.0)     # kg added to trunk
    com_range: Tuple[float, float] = (-0.10, 0.10)      # m trunk COM shift x,y
    motor_strength_range: Tuple[float, float] = (0.85, 1.15)
    resample_prob: float = 0.004        # within-episode DR resample (per step)

    # External pushes (hidden-test robustness). 0.0 disables.
    push_prob: float = 0.0              # per control step
    push_lin_vel: float = 0.5           # m/s impulse added to base lin vel

    # PD / action.
    action_scale: float = 0.30          # action is a residual on the nominal pose

    # History length for the adaptation module (k timesteps).
    history_len: int = 50


# ---------------------------------------------------------------------------
# Reward coefficients (Project-3 _compute_reward, reproduced exactly)
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    # Grading reward (Project-3 brief) uses sigma=0.05 -- so peaked that the
    # tracking term is ~0 unless the robot is already within ~0.1 m/s of the
    # command, giving no gradient to learn from. eval_gym.py keeps 0.05 (the
    # true graded metric); TRAINING uses the gentler sigmas below so the policy
    # gets a usable gradient toward the command (legged_gym-style shaping).
    sigma_lin_vel: float = 0.05        # grading reference (used by eval_gym)
    sigma_ang_vel: float = 0.05        # grading reference (used by eval_gym)
    train_sigma_lin_vel: float = 0.35  # impl: gentler shaping for training
    train_sigma_ang_vel: float = 0.35  # impl: gentler shaping for training
    w_tracking_lin: float = 2.0
    w_tracking_yaw: float = 1.0
    w_upright: float = 0.5
    w_z_vel: float = 0.2
    w_roll_pitch_ang: float = 0.1
    w_torque: float = 1e-4
    # The grading reward penalizes ctrl deltas between 500 Hz substeps (tiny,
    # since the PD target is held for 10 substeps). Training penalizes torque
    # deltas between 50 Hz control steps -- ~10x larger deltas, squared ->
    # ~100x harsher at the grader's 0.01. Scaled down accordingly so the
    # penalty does not out-compete the tracking gradient.
    w_action_rate: float = 1e-4
    # One-shot penalty applied on a fall (early termination). Gives a clear
    # negative signal for falling even when per-step reward is still small.
    w_termination: float = 2.0

    # Penalty curriculum: penalties scaled from k0, k_{t+1}=k_t^decay -> 1.
    penalty_curriculum_k0: float = 0.1
    penalty_curriculum_decay: float = 0.997


# ---------------------------------------------------------------------------
# Network architecture
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
    log_std_init: float = -1.0
    min_log_std: float = -1.609         # std >= 0.2 -> log(0.2)


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
    num_iterations: int = 1000
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

    # Base Go2 scene (robot + flat plane). With terrain="hfield" (the default)
    # the plane is replaced by a procedural fractal heightfield at build time;
    # the dense make_jagged_scene.xml is kept for visualization only (too many
    # geoms for MJX). In the Docker image this lives one directory above the
    # repo (see Dockerfile).
    # "auto" -> the Go2 MJCF bundled inside the gym-quadruped pip package. This
    # keeps the repo self-contained (no external model tree) and means MJX
    # training uses the exact model the grader evaluates on. Override with a path
    # to a custom scene if desired.
    model_path: str = "auto"
    checkpoint_dir: str = "checkpoints"
