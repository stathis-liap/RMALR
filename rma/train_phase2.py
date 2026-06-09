"""Entry point: Phase-2 adaptation-module training.

Usage:
    python -m rma.train_phase2 --phase1 checkpoints/phase1_final.pkl
"""
from __future__ import annotations

import argparse

from .config import Config
from .algos.adaptation import AdaptTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", type=str, default="checkpoints/phase1_final.pkl")
    ap.add_argument("--envs", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--terrain", type=str, default=None, choices=["scene", "hfield"])
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--ckpt-dir", type=str, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.envs is not None:
        cfg.adapt.num_envs = args.envs
    if args.iters is not None:
        cfg.adapt.num_iterations = args.iters
    if args.terrain is not None:
        cfg.env.terrain = args.terrain
    if args.model is not None:
        cfg.model_path = args.model
    if args.ckpt_dir is not None:
        cfg.checkpoint_dir = args.ckpt_dir

    print(f"[phase2] phase1={args.phase1} envs={cfg.adapt.num_envs} "
          f"iters={cfg.adapt.num_iterations}")
    trainer = AdaptTrainer(cfg, cfg.model_path, args.phase1)
    trainer.train()


if __name__ == "__main__":
    main()
