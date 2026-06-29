# RMA analysis & figure suite

Everything needed to **fully characterise the trained policy** in its three RMA
modes — `no_adapt` (lower bound), `rma` (the method), `expert` (upper bound) —
and to produce publication-quality figures that (a) reproduce and expand on the
RMA paper's figures and (b) answer every requirement of **Project 3:
Goal-Conditioned Go2 Velocity Tracking**.

```bash
source rma_venv/bin/activate

python -m analysis.make_figures --quick          # fast local pass (CPU, ~10–20 min)
python -m analysis.make_figures --full           # full quality (run on the GPU server)
python -m analysis.make_figures --only embedding adaptation --refresh
```

Figures are written to [`figures/`](../figures/); the raw rollout arrays are
cached in `figures/cache/*.npz`, so re-running only re-renders (instant) unless
you pass `--refresh`. `--quick` shrinks batch sizes/horizons for a laptop;
`--full` is sized for the GPU box (the rollouts are MJX, so they're seconds on a
GPU and minutes on CPU — same split as training).

## What runs the policy

All figures are built on one batched MJX rollout primitive
([`core.py`](core.py)) that runs the trained networks in the exact three
evaluation modes, identical to `rma.evaluate`:

| mode | extrinsics `z` | role | paper baseline |
|---|---|---|---|
| `no_adapt` | `μ(0)` | base policy, no env knowledge | Robust / *RMA w/o Adapt* (lower bound) |
| `rma` | `φ(history)` | adaptation module infers `z` online | **RMA** |
| `expert` | `μ(eₜ)` | sim's true privileged factors | **Expert** (upper bound) |

The MJX `Go2Env` mirrors the Project-3 proprioceptive observation, reward and PD
torque path used by the deployed `Controller`, and `randomize_models` gives
*per-env* friction/payload/COM — so a whole sweep axis collapses into one
batched rollout. The **faithful grader cross-check** (gym-quadruped, the actual
scoring environment) is `python -m rma.eval_gym` and `python -m testing.stress_test`.

## The figures

| # | file | what it shows | RMA paper | Project 3 requirement |
|---|---|---|---|---|
| 1 | `01_tracking_timeseries.png` | commanded vs achieved **vx, vy, yaw** through a scripted command sequence, 3 modes | — (we add it) | goal-conditioned tracking; *"smoothly adapt when the commanded velocity changes"* |
| 2 | `02_tracking_accuracy.png` | achieved vs commanded scatter over a command sweep; slope, R², in-tolerance % | — | velocity-tracking **accuracy** (the headline metric) |
| 3 | `03_tracking_polar.png` | omnidirectional tracking in the vx–vy plane (commanded circle vs achieved) | — | *vx, vy and yaw* commands across **all directions** |
| 4 | `04_generalization_sweeps.png` | survival + tracking error vs **friction / payload / motor strength**, 3 modes | **Fig. S4** | hidden tests: *friction variation, mass changes, actuator noise* |
| 5 | `05_stress_scenarios.png` | survival + tracking error per stress scenario, 3 modes | **Fig. 3** | hidden **stress tests**, external pushes, COM, combined |
| 6 | `06_summary_table.png` (+`.csv`/`.md`) | aggregate Success / TTF / reward / lin-err / yaw-err / torque / smoothness over the full DR distribution | **Table II** | overall benchmark performance + efficiency |
| 7 | `07_latent_embedding.png` | PCA of `z`: `φ(history)` recovers the same latent structure as `μ(eₜ)`, and they agree | expands Fig. 4 | evidence the *adaptation module* works |
| 8 | `08_adaptation.png` | `φ`'s extrinsics estimate converging to `μ`'s target (payload, friction) and tracking a **mid-episode actuator degradation** | **Fig. 4 / S3** | online adaptation to *disturbances / dynamics changes* |
| 9 | `09_hfield_terrain_sweep.png` | survival + error vs fractal terrain height **to 1.2 m (12× training)** on the MJX heightfield, 3 modes | **Fig. S4** (terrain) | hidden test: *terrain perturbations* (faithful, no leg-clipping) |
| 10 | `10_sensor_noise.png` | survival + error vs Gaussian sensor-noise std on `xₜ`, 3 modes | — | hidden test: *sensor noise* |
| 11 | `11_compound_difficulty.png` | survival + error vs a **compound** OOD difficulty (mass+COM+friction together): the bounds fan out `no_adapt < rma < expert` | — (conclusive bound demo) | required-comparison separation |
| 12 | `12_control_effort.png` | torque ⟨‖τ‖²⟩, roughness ⟨‖Δτ‖²⟩ and survival across nominal/slippery/ice/load/weak-motor | **Fig. 4** (torque) | *control-level* adaptation (efficient locomotion) |
| 13 | `13_load_bars_hfield.png` | **conclusive**: survival with **95% CIs** across increasing load **on the heightfield** (150 ep/cond), `no_adapt<rma<expert`, gap grows with load | **Fig. 3** (payload) | the adaptation **edge** (mass), statistically clean |
| 14 | `14_difficulty_hfield.png` | compound difficulty (mass+COM+friction) **on the heightfield** with 95% CIs | **Fig. S4** | bound ordering holds with terrain |

### Required-comparison coverage (Project 3)

The brief requires comparing the final controller against **a simple baseline**
and **a meaningful competing method**. This suite provides both as the bracketing
baselines that appear in *every* comparison figure (1, 4, 5, 6, 8):

* **Simple baseline** → `no_adapt`: the base policy with the extrinsics zeroed
  (`z = μ(0)`), i.e. a domain-randomisation policy with no online adaptation —
  the paper's *Robust* / *RMA-w/o-Adapt* lower bound.
* **Meaningful method** → `rma`: PPO base policy + **DAgger**-trained adaptation
  module (`φ`), evaluated against the privileged **`expert`** oracle upper bound.

## Reading the figures

* **survival** is the primary safety metric (% of envs that never fell);
  **lin-err / yaw-err** are the velocity / yaw-rate tracking errors (m/s, rad/s,
  lower = better). Compare on **survival first, then tracking error**.
* The expected ordering everywhere is **`no_adapt ≤ rma ≤ expert`**. Where the
  three coincide, the base policy is simply robust enough there that knowing `eₜ`
  doesn't help — itself a valid finding (the paper sees the same on easy regimes).
* On the sweeps, the shaded band is the **training range**; everything outside it
  is out-of-distribution generalisation.

## Notes

* These rollouts use the MJX training simulator (for batched access to `z`, the
  true `eₜ`, contacts and per-env factors). Headline numbers should be confirmed
  in the **grader** with `python -m rma.eval_gym` — the Controller there consumes
  only the benchmark's proprioceptive observations.
* `expert` here is the *true* oracle: `eₜ` is read straight from the sim.
* The stress-scenario presets are shared with `testing/scenarios.py`; terrain
  *scenes* (perlin/boxes) are gym-quadruped-only and reduce to flat here, so the
  bar chart focuses on the physics stressors (friction/payload/COM/motor/pushes).
  To stress on the actual training **heightfield**, use
  `python -m testing.stress_test --hfield`.
