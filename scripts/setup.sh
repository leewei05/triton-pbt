#!/bin/bash

module load miniforge3
module load cuda/12.6.3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate triton_env
