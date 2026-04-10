#!/bin/bash
# setup.sh: Load modules and activate the virtual environment

# 1. Load system-optimized Python and CUDA
module load python/3.10.3
module load cuda/12.6.3
module load uv
module load deeplearning/2025.4
module load py-pytest/7.1.3

# 2. Activate the venv
# Assuming the venv is created in your home directory
if [ -d "$HOME/triton_env" ]; then
    source "$HOME/triton_env/bin/activate"
    echo "✅ venv 'triton_env' active with CUDA 12.6.3"
else
    echo "❌ Error: triton_env not found. Run init.sh first."
fi