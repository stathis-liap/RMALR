# Stress-testing the trained RMA policy

A configurable harness that pushes the trained Go2 policy to its limits across
out-of-distribution scenarios — rough terrain, extreme friction, payloads, COM
shifts, weak motors, and periodic shoves — and reports **survival** and
**velocity-tracking accuracy**.

It drives the exact deployment path used for grading (`gym-quadruped` +
`rma.controller.Controller`), so results are faithful. **CPU-only — runs on your
laptop.** Make sure the venv is active:

```bash
source rma_venv/bin/activate
```

## Quick start

```bash
# one scenario
python -m testing.stress_test --scenario slippery --episodes 20 \
    --phase1 checkpoints/phase1_final.pkl --phase2 checkpoints/phase2_final.pkl

# the whole suite, printed as a summary table
python -m testing.stress_test --scenario all --episodes 20 \
    --phase1 checkpoints/phase1_final.pkl --phase2 checkpoints/phase2_final.pkl

# list the presets
python -m testing.stress_test --list
```

## Stress on the training terrain (`--hfield`)

By default the scenarios run on gym-quadruped (flat / perlin / boxes). Add
`--hfield` to instead run them on the **MJX training heightfield** — the gentle,
walkable terrain the policy actually learned on (gym-quadruped's `perlin` is a
0.56 m torture field where the legs clip through). The scenario's stressors
(friction, payload, COM, motor strength, pushes) are pinned into the MJX env's
domain randomization, and a *batch* of envs is rolled out in parallel:

```bash
python -m testing.stress_test --scenario all --hfield --episodes 256 \
    --phase1 checkpoints/phase1_final.pkl --phase2 checkpoints/phase2_final.pkl
```

Notes:
- `--episodes` is the number of **parallel envs** here (use 128–512 for good
  statistics — they run together, not one-by-one).
- This is a **batched MJX** rollout: fast on a GPU, slow to compile on CPU
  (~30–45 s per scenario). Run it on the training server for the full suite.
- `--hfield-z` sets terrain height (default `0.10` = training).
- The `scene` of each preset is ignored (terrain is always the hfield), so
  `rough_perlin`/`boxes` reduce to `nominal`; the other stressors still apply
  (e.g. `gauntlet` = low friction + payload + pushes + weak motors on terrain).
- `expert` here is the *true* oracle (the MJX `e_t` carries the real terrain
  height), so `--mode all` gives a clean `no_adapt ≤ rma ≤ expert` on terrain.

## Watch it (on-screen viewer)

Needs a display; use the GLFW backend and fewer episodes:

```bash
MUJOCO_GL=glfw python -m testing.stress_test --scenario gauntlet --episodes 3 --render \
    --phase1 checkpoints/phase1_final.pkl --phase2 checkpoints/phase2_final.pkl
```

## Manual teleop (drive it with the keyboard)

Steer the policy yourself in a live viewer — the typed velocity command is fed
straight to the controller:

```bash
MUJOCO_GL=glfw python -m testing.manual --scene flat \
    --phase1 checkpoints/phase1_final.pkl --phase2 checkpoints/phase2_final.pkl
```

Keys (focus the viewer window):

| key | action |
|---|---|
| `I` / `K` | forward / backward |
| `J` / `L` | strafe left / right |
| `W` / `S` | increase / decrease speed |
| `A` / `D` | turn left / right (yaw) |
| `X` | stop (zero the command) |
| `R` | reset the episode |
| close window | quit |

Movement is on `I`/`J`/`K`/`L` (not the arrows): gym-quadruped's viewer already
binds the arrow keys, space, and ctrl for its own use, so we stay off them.
The current command is printed in the terminal. Forward + `A`/`D` walks a curve.
Try `--scene perlin` or `--scene random_boxes` for rough ground, or `--mode
no_adapt` to feel the difference without the adaptation module.

## See it on the smooth training terrain

Use this for terrain, **not** gym-quadruped's `perlin` scene: that one is a
**0.56 m** jagged field — taller than the 0.28 m robot — so on it the legs
visually clip through the over-tall terrain (the robot still walks; it's not
frozen, it just *looks* buried). To watch the policy on the gentle, walkable
heightfield it actually trained on, render the MJX training env directly:

```bash
MUJOCO_GL=glfw python -m testing.view_hfield \
    --phase1 checkpoints/phase1_final.pkl --phase2 checkpoints/phase2_final.pkl
```

The terrain is a textured, gently rolling heightfield (you can clearly see the
undulation) and the robot **walks autonomously** — the env issues random velocity
commands and resamples them every couple of seconds (the current command prints
in the terminal). The camera follows the robot, which respawns only on a real
fall (no episode timeout). For keyboard teleop instead, use `testing.manual`.

Options: `--z-scale` sets terrain height (default `0.12`; `0.10` = exact
training, raise to roughen — it stays walkable well past `0.2`), and
`--mode {rma,no_adapt,expert}` (here `expert` is the *true* oracle — the MJX
`e_t` carries the real local terrain height, unlike the flat-only gym expert).

## Lower bound, RMA, upper bound (`--mode all`)

The three modes bracket the policy, exactly like the paper's baselines:

| mode | extrinsics `z` | meaning |
|---|---|---|
| `no_adapt` | `μ(0)` | base policy, no env info — **lower bound** |
| `rma` | `φ(history)` | adaptation module estimates `z` from history |
| `expert` | `μ(e_t)` | the sim's **true** privileged factors — **upper bound** |

`--mode all` runs all three on each scenario and prints them together:

```bash
python -m testing.stress_test --scenario all --mode all --episodes 20 \
    --phase1 checkpoints/phase1_final.pkl --phase2 checkpoints/phase2_final.pkl
```

What to look for: **`no_adapt ≤ rma ≤ expert`** on survival + tracking error.
RMA should close the gap toward expert wherever the disturbance is inferable from
history (added mass, friction). If `rma ≈ no_adapt ≈ expert`, the base policy is
simply robust enough there that knowing `e_t` doesn't help — also a valid finding.

The `expert` mode reads the true `e_t` straight from the sim after each reset
(trunk mass, COM, foot friction, motor scale; terrain height is left 0 on flat).

## Ad-hoc scenarios

Override any stressor from the command line — applied on top of the chosen
preset (default `nominal`), so you can dial in a custom test without editing code:

```bash
python -m testing.stress_test --scene perlin --friction 0.25 \
    --push-vel 0.6 --push-interval 100 --episodes 10 \
    --phase1 checkpoints/phase1_final.pkl --phase2 checkpoints/phase2_final.pkl
```

Override flags: `--scene {flat,perlin,random_boxes}`, `--friction`, `--payload`,
`--com-x`, `--motor-scale`, `--push-vel`, `--push-interval`.

## The presets

| scenario | what it stresses |
|---|---|
| `nominal` | baseline (flat, in-distribution friction) |
| `slippery` / `ice` | low / very low friction (0.3 / 0.15) |
| `sticky` | high friction (2.5) |
| `payload_5kg` / `payload_10kg` | added trunk mass (10 kg ≈ body weight) |
| `front_heavy` | COM shifted 10 cm forward + 3 kg |
| `weak_motors` / `very_weak_motors` | torque scaled to 80% / 65% |
| `pushes_mild` / `pushes_hard` | periodic base-velocity shoves |
| `rough_perlin` | rough perlin heightfield |
| `boxes` | field of random boxes |
| `gauntlet` | everything at once |

## Reading the results

- **survival** — % of episodes that finished without falling. The primary metric.
- **lin_err / yaw_err** — mean velocity / yaw-rate tracking error (m/s, rad/s)
  over the episode. Lower is better.
- **steps** — mean episode length (capped at `--max-steps`, default 2000).

Compare scenarios on **survival first, then tracking error** — not on reward
(a long survival accumulates penalty terms, so reward couples with episode
length and is misleading).

## Notes

- Each scenario's stressors are independent and compose freely (see `gauntlet`).
  Add your own in [`scenarios.py`](scenarios.py).
- The perlin/box terrain uses gym-quadruped's procedural generator (fixed seed),
  so the terrain is the same each episode while the command/heading varies.
- `random_pyramids` is intentionally omitted — it crashes inside gym-quadruped's
  own generator (upstream bug), unrelated to this harness.
