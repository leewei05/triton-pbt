#!/bin/bash

module load miniforge3
mamba create -n triton_env python=3.10 -y

# Use the setup script to activate and load CUDA
source ./scripts/setup.sh

# Install heavy hitters
pip install torch triton hypothesis pytest
