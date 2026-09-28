# Training on a GPU

Wild (and every JAX world after it) runs unchanged on an NVIDIA GPU, typically 50–100×
faster than on a few CPU cores. Town Life is numpy and uses CPU cores instead.

## Your own PC (e.g. RTX 3080)
* Linux: an NVIDIA driver (check with `nvidia-smi`), Python 3.11+.
* Windows: JAX's GPU builds are Linux-only, so use WSL2 (`wsl --install -d Ubuntu`); the
  Windows NVIDIA driver is used from inside WSL, install nothing CUDA-related there.

```bash
git clone https://github.com/neomorrison/humanoid-civilization && cd humanoid-civilization
bash scripts/gpu_setup.sh             # installs JAX with CUDA, checks it sees the GPU
bash scripts/gpu_run.sh wild 600      # 10 hours; pushes logs to a results/... branch every 15 min
```
10–12 GB of GPU memory fits hundreds of worlds with raw senses. Storage stays, so this is
the best place for long runs.

## NRP Nautilus JupyterHub
Pick **NRP Deep Learning & Data Science Full, PyTorch (CUDA)** with 1 GPU, open a
terminal, and run the same commands. Pods and their storage are removed, so always use
`gpu_run.sh` (it pushes results to git as it goes) and expect to restart long runs. Good
for running several experiments side by side; the 16 CPU cores also train Town Life
about 4× faster than a 4-core machine (`bash scripts/gpu_run.sh town 600`).

Pushing needs git credentials on that machine (a GitHub token or SSH key with access to
this repository).
