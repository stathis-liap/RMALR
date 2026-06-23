# RMA for Go2 — Rapid Motor Adaptation (Project 3: velocity tracking)

An implementation of **RMA: Rapid Motor Adaptation for Legged Robots**
(Kumar et al., 2021) for the Unitree **Go2**, adapted to the goal-conditioned
**velocity-tracking** task of the `gym-quadruped` benchmark.

The policy is trained entirely in simulation (**MJX**, on GPU) and deployed
through a thin `Controller` that runs the trained networks inside the
`gym-quadruped` environment used for grading.

---

## How it works

RMA has two subsystems trained in two phases (paper Fig. 2):

* **Base policy π + environment-factor encoder μ** (Phase 1). Trained with PPO.
  At each step π sees the proprioceptive state `x_t`, the previous action, and
  the *extrinsics* `z_t = μ(e_t)`, where `e_t` is **privileged** info available
  only in simulation (payload, COM shift, per-motor strength, friction, local
  terrain height). μ compresses those 17 numbers into an 8-d latent `z_t`.

* **Adaptation module φ** (Phase 2). At deployment we don't have `e_t`. φ is a
  1-D CNN over the last 50 steps of state+action history; it estimates
  `ẑ_t ≈ z_t` so π keeps working without privileged info. Trained by supervised
  regression to `μ(e_t)` on **on-policy** (DAgger) rollouts of π driven by φ's
  own prediction.

At deployment π runs every control step and φ refreshes `ẑ_t` asynchronously
(~10 Hz), exactly as in the paper.

**Key implementation details**

| Item | Value |
|---|---|
| Policy state `x_t` (36) | `gravity_base(3) · base_ang_vel(3) · qpos(12) · qvel(12) · lin_vel_err(3) · ang_vel_err(3)` |
| Privileged `e_t` (17) | `payload(1) · com_xy(2) · motor_strength(12) · friction(1) · terrain_height(1)` |
| Control | residual joint target → PD torque (Kp/Kd randomized), 50 Hz policy / 500 Hz sim |
| Terrain | procedural **fractal heightfield** (`terrain="hfield"`); per-env random spawn samples different patches |
| Reward | Project-3 velocity-tracking reward + gait shaping (air-time, foot clearance, hip width, anti-slip) |

The velocity command enters `x_t` as the base-frame tracking errors the
benchmark provides, so the policy is goal-conditioned on `[vx, vy, yaw_rate]`.

---

## Repository layout

```
rma/
  config.py              # all hyperparameters (one place to tune)
  controller.py          # deployment Controller for the gym-quadruped benchmark
  train_phase1.py        # entry point: Phase-1 PPO (π + μ)
  train_phase2.py        # entry point: Phase-2 adaptation module (φ)
  eval_gym.py            # evaluate inside the gym-quadruped grader (survival, tracking)
  evaluate.py            # quick in-sim (MJX) sanity eval
  envs/
    go2_env.py           # MJX environment (reward, termination, terrain, DR)
    build_model.py       # builds the Go2 MjModel (heightfield, foot-only collisions)
    terrain.py           # fractal heightfield generator
    go2_constants.py     # Go2 joint ordering / model resolution
  models/networks.py     # μ, π, value, φ (Flax)
  algos/ppo.py           # Phase-1 PPO trainer
  algos/adaptation.py    # Phase-2 DAgger trainer
tools/
  view_policy.py         # watch a checkpoint drive the Go2 (on-screen viewer)
Dockerfile               # GPU training image (self-contained)
```

---

## Prerequisites

* **Training server**: NVIDIA GPU + Docker with the NVIDIA container runtime
  (`--gpus all`). Nothing else to install — the image is self-contained.
* **Local PC**: Python 3.10+ for evaluation / visualization (CPU is fine).
  No external model files are needed; the Go2 model ships inside `gym-quadruped`.

Server location used below (edit to taste):

```bash
SERVER=stathisliap@150.140.184.61      # the GPU training host
REMOTE=~/rmalr                          # project dir on the server
```

---

## A. On the server — training (GPU, Docker)

### 1. Sync the code to the server

From your local repo root:

```bash
rsync -av --exclude rma_venv --exclude checkpoints --exclude '.git' \
      ./ $SERVER:$REMOTE/
# or, for a couple of changed files:
# scp rma/config.py rma/envs/go2_env.py $SERVER:$REMOTE/rma/...
```

### 2. Build the Docker image (once, and after dependency changes)

```bash
ssh $SERVER
cd $REMOTE
docker build -t rma-go2 .
```

The build runs a sanity check that the model + networks wire up, so a broken
dependency fails the build early.

### 3. Phase 1 — train the base policy π + encoder μ

Checkpoints are written to `checkpoints/` on the host (bind-mounted), so they
survive the container:

```bash
docker run --gpus all -it --rm \
    -v $PWD/checkpoints:/app/checkpoints \
    rma-go2 \
    python -m rma.train_phase1
```

Produces `checkpoints/phase1_<iter>.pkl` periodically and
`checkpoints/phase1_final.pkl` at the end. Trains on the heightfield by default.

Useful overrides (all optional):

```bash
    python -m rma.train_phase1 --envs 4096 --iters 15000 --terrain hfield
    #   --terrain scene   # flat ground instead of the heightfield
    #   --ckpt-dir DIR    # write checkpoints somewhere else
```

### 4. Phase 2 — train the adaptation module φ

Needs a finished Phase-1 checkpoint:

```bash
docker run --gpus all -it --rm \
    -v $PWD/checkpoints:/app/checkpoints \
    rma-go2 \
    python -m rma.train_phase2 --phase1 checkpoints/phase1_final.pkl
```

Produces `checkpoints/phase2_<iter>.pkl` and `checkpoints/phase2_final.pkl`.
Runs for 2000 iterations by default (`--iters` to change).

> Tip: run long jobs detached (`docker run -d ...` or inside `tmux`) so they
> survive an SSH disconnect.

### 5. See the logs (TensorBoard)

Both trainers log scalars to `checkpoints/tb/`. Start TensorBoard **on the
server** in a container:

```bash
docker run -d --rm \
    -v $PWD/checkpoints:/app/checkpoints \
    -p 6006:6006 --name rma_tb rma-go2 \
    tensorboard --logdir /app/checkpoints/tb --host 0.0.0.0 --port 6006
```

Then, **from your local PC**, open an SSH tunnel and browse it:

```bash
ssh -L 6006:localhost:6006 $SERVER
# leave that open, then visit http://localhost:6006 in your browser
```

Watch `reward/tracking_lin`, `reward/tracking_yaw`, `episode/length_est`, and
`policy/std` for Phase 1; `train/mse` for Phase 2. (Stop the viewer later with
`docker stop rma_tb`.)

---

## B. On the local PC — download, evaluate, visualize (CPU)

### 1. One-time local setup

```bash
python3 -m venv rma_venv
source rma_venv/bin/activate
pip install -r requirements.txt          # CPU jax is fine for eval/viz
```

### 2. Download the trained checkpoints from the server

```bash
mkdir -p checkpoints
scp $SERVER:$REMOTE/checkpoints/phase1_final.pkl checkpoints/
scp $SERVER:$REMOTE/checkpoints/phase2_final.pkl checkpoints/
# or grab everything:
# rsync -av $SERVER:$REMOTE/checkpoints/ ./checkpoints/
```

### 3. Watch the policy walk (on-screen viewer)

```bash
source rma_venv/bin/activate

# full RMA (base policy + adaptation module):
python tools/view_policy.py \
    --phase1 checkpoints/phase1_final.pkl \
    --phase2 checkpoints/phase2_final.pkl --mode rma

# base policy only (no adaptation), e.g. to inspect a Phase-1-only checkpoint:
python tools/view_policy.py --phase1 checkpoints/phase1_final.pkl
```

Close the viewer window to stop. (Needs a display; use the venv, not Docker.)

### 4. Score it in the benchmark (survival + tracking error)

```bash
python -m rma.eval_gym \
    --phase1 checkpoints/phase1_final.pkl \
    --phase2 checkpoints/phase2_final.pkl \
    --mode rma --episodes 50
```

Reports survival rate, episode reward, and velocity / yaw-rate tracking error.
Run with `--mode no_adapt` to compare against the base policy with `z = μ(0)`.
Domain-shift sweeps are supported, e.g. `--friction 0.3` or `--payload 5`.

> **Metric note:** compare methods on **survival + tracking error**, not mean
> reward — surviving longer with poor tracking accumulates more reward than
> falling early, so reward couples with episode length and is misleading.

---

## Where to change things

Everything tunable lives in [`rma/config.py`](rma/config.py):

* **Terrain** — `EnvConfig.terrain` (`"hfield"` / `"scene"`), `fractal_z_scale`
  (terrain height), `hfield_radius`, `fractal_base_freq`.
* **Velocity-tracking priority** — `RewardConfig.w_tracking_lin/_yaw` and
  `train_sigma_lin_vel/_ang_vel` (smaller sigma = stricter accuracy).
* **Gait** — `foot_clearance_target` (how high the feet swing),
  `air_time_target`, `w_foot_slip`, `w_hip_deviation`.
* **Domain randomization** — `friction_range`, `kp_range`, `kd_range`,
  `payload_range`, `com_range`, `motor_strength_range`.
* **Training length** — `PPOConfig.num_iterations` (Phase 1),
  `AdaptConfig.num_iterations` (Phase 2).
