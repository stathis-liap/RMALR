"""Stress-testing harness for the trained RMA Go2 policy.

Drives the deployment path (gym-quadruped + ``rma.controller.Controller``) through
a suite of out-of-distribution scenarios -- rough terrain, low/high friction,
payloads, COM shifts, weak motors, and periodic shoves -- and reports survival
and velocity-tracking accuracy. CPU-only; runs on a laptop.

CLI: ``python -m testing.stress_test --help``
"""
