# RMA on MuJoCo/MJX for the Go2 velocity-tracking benchmark (Project 3).
#
# Build:  docker build -t rma-go2 .
#
# TRAINING loads the base Go2 scene (robot + plane) and injects a procedural
# MJX heightfield (terrain="hfield" in rma/config.py). That base scene lives one
# directory above the repo at ../unitree_mujoco/unitree_robots/go2/scene.xml.
# Mount the model tree so it resolves to /unitree_mujoco (the parent of /app):
#
#   docker run --gpus all -it --rm \
#       -v $PWD/checkpoints:/app/checkpoints \
#       -v $(realpath ../unitree_mujoco):/unitree_mujoco \
#       rma-go2 python -m rma.train_phase1 --envs 4096
#
# Evaluation in the official gym-quadruped benchmark uses the bundled Go2 model
# (CPU MuJoCo), no mount required:
#   docker run -it --rm -v $PWD/checkpoints:/app/checkpoints rma-go2 \
#       python -m rma.eval_gym --episodes 20
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

# JAX with CUDA 12 (pulls matching jaxlib). Pin if you need reproducibility.
RUN pip install "jax[cuda12]>=0.4.34"

COPY requirements.txt .
# jax/jaxlib already installed above; install the rest (incl. gym-quadruped).
RUN pip install mujoco>=3.2.0 mujoco-mjx>=3.2.0 flax>=0.8.0 optax>=0.2.0 numpy>=1.26 \
                "gym-quadruped>=1.1.2" "gymnasium>=1.0" scipy>=1.11 matplotlib>=3.7 \
                imageio>=2.34 imageio-ffmpeg>=0.4

COPY . .

# Regenerate the jagged Go2 scene if the model tree is mounted at build time.
# (Usually you mount ../unitree_mujoco at run time instead -- see header.)
RUN python tools/make_jagged_scene.py \
        --out ../unitree_mujoco/unitree_robots/go2/scene_jagged.xml \
    || echo "go2 model tree not present at build; mount it at run time"

CMD ["python", "-m", "rma.train_phase1", "--envs", "4096"]
