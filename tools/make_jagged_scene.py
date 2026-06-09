"""Generate a jagged-terrain MuJoCo scene for the Unitree Go2.

Takes the stock `go2/scene.xml` layout and replaces the flat plane with a
challenging surface made of:
  * a dense field of small boxes with randomized heights (rough / jagged ground)
  * a flat "spawn pad" at the origin so the robot starts on stable ground
  * an escalating staircase section (kept from the user's example scene)

The robot model itself is unchanged (`<include file="go2.xml"/>`), so this file
drops straight into unitree_mujoco/unitree_robots/go2/.

Run:
    python tools/make_jagged_scene.py \
        --out ../unitree_mujoco/unitree_robots/go2/scene_jagged.xml
"""
from __future__ import annotations

import argparse
import random


HEADER = """<mujoco model="go2 jagged scene">
  <include file="go2.xml"/>

  <statistic center="0 0 0.1" extent="0.8"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="rough" rgba="0.45 0.40 0.35 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
"""

FOOTER = """  </worldbody>
</mujoco>
"""


def jagged_field(
    x_range=(-4.0, 4.0),
    y_range=(-4.0, 4.0),
    tile=0.30,
    max_height=0.09,
    spawn_radius=0.6,
    seed=0,
):
    """Grid of boxes with random heights; flat pad cleared near the origin."""
    rng = random.Random(seed)
    half = tile / 2.0
    geoms = []
    x = x_range[0]
    while x <= x_range[1] + 1e-6:
        y = y_range[0]
        while y <= y_range[1] + 1e-6:
            # leave a flat spawn pad around the origin
            if (x * x + y * y) ** 0.5 < spawn_radius:
                y += tile
                continue
            h = rng.uniform(0.0, max_height)
            sz = max(h / 2.0, 0.005)
            cz = sz
            geoms.append(
                f'    <geom type="box" material="rough" '
                f'pos="{x:.3f} {y:.3f} {cz:.4f}" '
                f'size="{half:.3f} {half:.3f} {sz:.4f}"/>'
            )
            y += tile
        x += tile
    return geoms


def staircase(x0=5.0, y0=0.0, steps=8, rise=0.06, run=0.30, width=2.0):
    """An ascending staircase (extra challenge, mirrors the example scene)."""
    geoms = []
    for i in range(steps):
        h = rise * (i + 1)
        cz = h / 2.0
        cx = x0 + run * i
        geoms.append(
            f'    <geom type="box" material="rough" '
            f'pos="{cx:.3f} {y0:.3f} {cz:.4f}" '
            f'size="{run/2:.3f} {width/2:.3f} {h/2:.4f}"/>'
        )
    return geoms


def build(seed=0):
    parts = [HEADER]
    parts.append("    <!-- jagged rough-terrain field -->")
    parts += jagged_field(seed=seed)
    parts.append("    <!-- ascending staircase -->")
    parts += staircase()
    parts.append(FOOTER)
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", type=str,
        default="../unitree_mujoco/unitree_robots/go2/scene_jagged.xml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    xml = build(args.seed)
    with open(args.out, "w") as f:
        f.write(xml)
    n = xml.count("<geom")
    print(f"wrote {args.out} ({n} geoms, seed={args.seed})")


if __name__ == "__main__":
    main()
