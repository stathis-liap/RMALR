"""Entry point: Phase-1 PPO (base policy + env-factor encoder).

Usage:
    python -m rma.train_phase1 [--envs N] [--iters N] [--terrain flat|hfield]
"""
from __future__ import annotations

import argparse

from .config import Config
from .algos.ppo import PPOTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--terrain", type=str, default=None, choices=["scene", "hfield"])
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--ckpt-dir", type=str, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.envs is not None:
        cfg.ppo.num_envs = args.envs
    if args.iters is not None:
        cfg.ppo.num_iterations = args.iters
    if args.terrain is not None:
        cfg.env.terrain = args.terrain
    if args.model is not None:
        cfg.model_path = args.model
    if args.ckpt_dir is not None:
        cfg.checkpoint_dir = args.ckpt_dir

    print(f"[phase1] envs={cfg.ppo.num_envs} iters={cfg.ppo.num_iterations} "
          f"terrain={cfg.env.terrain}")
    trainer = PPOTrainer(cfg, cfg.model_path)
    trainer.train()


if __name__ == "__main__":
    main()
