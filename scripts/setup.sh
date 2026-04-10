#!/bin/bash
# setup.sh: Load modules and activate the virtual environment

set -x

# 1. Load system-optimized Python and CUDA
module load python/3.10.3
module load cuda/12.6.3
module load uv

# 2. Activate the venv
# Assuming the venv is created in your home directory
if [ -d "$HOME/triton_env" ]; then
    source "$HOME/triton_env/bin/activate"
    echo "✅ venv 'triton_env' active with CUDA 12.6.3"
else
    echo "❌ Error: triton_env not found. Run init.sh first."
fi