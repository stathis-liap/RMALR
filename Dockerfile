# RMA on MuJoCo/MJX for the Go2 velocity-tracking benchmark (Project 3).
#
# This image is fully self-contained: both MJX training and the gym-quadruped
# evaluation use the Go2 model that ships inside the gym-quadruped pip package
# (rma/config.py: model_path="auto"). No external model tree or mount is needed.
#
# Build:
#   docker build -t rma-go2 .
#
# Train (persist checkpoints to the host):
#   docker run --gpus all -it --rm -v $PWD/checkpoints:/app/checkpoints rma-go2 \
#       python -m rma.train_phase1
#   docker run --gpus all -it --rm -v $PWD/checkpoints:/app/checkpoints rma-go2 \
#       python -m rma.train_phase2 --phase1 checkpoints/phase1_final.pkl
#
# Evaluate in the gym-quadruped benchmark (CPU is fine for eval):
#   docker run -it --rm -v $PWD/checkpoints:/app/checkpoints rma-go2 \
#       python -m rma.eval_gym --phase1 checkpoints/phase1_final.pkl \
#                              --phase2 checkpoints/phase2_final.pkl --episodes 20
#
# TensorBoard (trainers write to checkpoints/tb/). On the server:
#   docker run -d --rm -v $PWD/checkpoints:/app/checkpoints -p 6006:6006 \
#       --name rma_tb rma-go2 \
#       tensorboard --logdir /app/checkpoints/tb --host 0.0.0.0 --port 6006
# Then from your local PC:  ssh -L 6006:localhost:6006 <user>@<server>
# and open http://localhost:6006
#
# Base: CUDA 12 + cuDNN runtime on Ubuntu 22.04 (matches jax[cuda12] wheels).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # MJX/EGL headless rendering
    MUJOCO_GL=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip git \
        libgl1 libglew2.2 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && python -m pip install --upgrade pip

WORKDIR /app

# JAX with CUDA 12 first (pulls matching jaxlib); the rest must not downgrade it.
RUN pip install "jax[cuda12]>=0.4.34"

# System deps for gym-quadruped, placed after the JAX layer so editing them
# doesn't invalidate the slow jax[cuda12] install cache:
#   * build-essential + python3.10-dev : compile the `noise` C extension
#   * libglib2.0-0 / libsm6 / libxext6 / libxrender1 : OpenCV (cv2) runtime libs
#     (cv2 is imported by gym_quadruped.utils.mujoco.terrain; needs libgthread)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3.10-dev \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install everything else (incl. gym-quadruped, which bundles the Go2 model).
RUN pip install mujoco>=3.2.0 mujoco-mjx>=3.2.0 flax>=0.8.0 optax>=0.2.0 numpy>=1.26 \
                "gym-quadruped>=1.1.2" "gymnasium>=1.0" scipy>=1.11 matplotlib>=3.7 \
                tensorboardX>=2.6 tensorboard>=2.15 \
                imageio>=2.34 imageio-ffmpeg>=0.4

COPY . .

# Sanity-check at build time that the model + networks wire up (fails the build
# early if a dependency is missing). Does not need a GPU.
RUN python -c "from rma.config import Config; from rma.envs.build_model import build_model; \
m=build_model(Config().env, Config().model_path); print('build OK:', m.ngeom, 'geoms')"

# Default: launch Phase-1 training. Override with eval_gym / train_phase2 as above.
CMD ["python", "-m", "rma.train_phase1"]
