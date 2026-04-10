#!/bin/bash

module load miniforge3

rm -rf ~/.conda/envs/triton_env

conda config --set solver libmamba
conda config --set channel_priority strict
conda create -n triton_env python=3.10 -y

# Use the setup script to activate and load CUDA
source ./scripts/setup.sh

# Install heavy hitters
pip install torch triton hypothesis pytest
