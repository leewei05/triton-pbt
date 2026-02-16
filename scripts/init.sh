#!/bin/bash

module load miniforge3
mamba create -n triton_env python=3.10 -y
conda activate triton_env
# Install your heavy hitters once
pip install torch triton
