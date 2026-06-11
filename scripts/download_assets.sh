#!/usr/bin/env bash
# Asset setup for RMA Go2 training/eval.
#
# Nothing to download: both MJX training and the gym-quadruped evaluation use the
# Go2 model that ships inside the gym-quadruped pip package (config model_path
# defaults to "auto" -> that bundled model). Just install the requirements.
#
# This script simply verifies the model is importable so you can fail fast.
set -euo pipefail

python - <<'PY'
from rma.envs.go2_constants import package_go2_path
print("[assets] gym-quadruped Go2 model:", package_go2_path())
print("[assets] OK -- no external assets required.")
PY
