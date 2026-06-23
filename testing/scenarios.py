"""Stress-test scenario definitions and a registry of presets.

A ``Scenario`` is a bundle of stressors applied to the gym-quadruped deployment
env. Each one is independent, so presets compose them freely (see ``gauntlet``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Scenario:
    name: str
    description: str = ""
    # Terrain: one of "flat", "perlin" (rough heightfield), "random_boxes".
    scene: str = "flat"
    # Ground friction sampled per episode in [lo, hi]; pass (f, f) to pin it.
    friction: Tuple[float, float] = (0.4, 1.2)
    payload: float = 0.0               # kg added to the trunk
    com_shift: Tuple[float, float] = (0.0, 0.0)   # m, trunk COM xy offset
    motor_scale: float = 1.0           # multiply applied torque (<1 = weak motors)
    push_vel: float = 0.0              # m/s base-velocity impulse per shove
    push_interval: int = 0             # env steps between shoves (0 = no pushes)
    push_axes: Tuple[str, ...] = ("x", "y")   # horizontal axes a shove can hit


# Ordered roughly easy -> hard; `all` runs them in this order.
_PRESETS = [
    Scenario("nominal", "baseline: flat ground, in-distribution friction"),
    Scenario("slippery", "low friction (0.3)", friction=(0.3, 0.3)),
    Scenario("ice", "very low friction (0.15)", friction=(0.15, 0.15)),
    Scenario("sticky", "high friction (2.5)", friction=(2.5, 2.5)),
    Scenario("payload_5kg", "+5 kg on the trunk", payload=5.0),
    Scenario("payload_10kg", "+10 kg (~body weight)", payload=10.0),
    Scenario("front_heavy", "COM shifted forward 10 cm + 3 kg",
             payload=3.0, com_shift=(0.10, 0.0)),
    Scenario("weak_motors", "torque scaled to 80%", motor_scale=0.80),
    Scenario("very_weak_motors", "torque scaled to 65%", motor_scale=0.65),
    Scenario("pushes_mild", "0.4 m/s shove every 150 steps",
             push_vel=0.4, push_interval=150),
    Scenario("pushes_hard", "0.8 m/s shove every 100 steps",
             push_vel=0.8, push_interval=100),
    Scenario("rough_perlin", "rough perlin heightfield", scene="perlin"),
    Scenario("boxes", "field of random boxes", scene="random_boxes"),
    Scenario("gauntlet", "everything at once: perlin + low friction + payload "
             "+ shoves + weak motors", scene="perlin", friction=(0.3, 0.3),
             payload=5.0, motor_scale=0.85, push_vel=0.5, push_interval=120),
]

SCENARIOS = {s.name: s for s in _PRESETS}
