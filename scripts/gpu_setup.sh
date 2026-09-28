#!/usr/bin/env bash
# Set up a GPU machine for training (Linux or WSL2 with an NVIDIA driver; see docs/GPU.md).
#   bash scripts/gpu_setup.sh            # installs JAX with CUDA and checks that it sees the GPU
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt "jax[cuda12]==0.10.2"
python3 - <<'PY'
import time, jax, jax.numpy as jnp
print("JAX", jax.__version__, "devices:", jax.devices())
assert jax.devices()[0].platform == "gpu", "JAX does not see a GPU (check the driver: nvidia-smi)"
x = jnp.ones((4096, 4096)); (x @ x).block_until_ready(); t = time.time()
for _ in range(10): (x @ x).block_until_ready()
print(f"matmul: {10 * 2 * 4096**3 / (time.time() - t) / 1e12:.1f} TFLOP/s")
PY
