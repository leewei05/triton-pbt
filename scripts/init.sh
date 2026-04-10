#!/bin/bash
# init.sh: Create the environment from scratch

set -x

# 1. Clean up old remnants (Conda or venv)
rm -rf ~/.conda/envs/triton_env
rm -rf "$HOME/triton_env"

# 2. Load Python and create the venv
module load python/3.10.3
python -m venv "$HOME/triton_env" --without-pip

# 3. Use setup.sh to activate
source ./scripts/setup.sh

# 4. Install heavy hitters using uv
# This bypasses the slow metadata solving that killed Conda
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_INSTALLS=1
uv pip install torch triton hypothesis pytest

echo "🚀 Project initialized successfully with venv."
