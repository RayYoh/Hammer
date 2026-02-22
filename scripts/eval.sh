#!/bin/bash

# Usage: bash scripts/eval.sh

# Set the used cuda path; maybe unnecessary
export PATH=$HOME/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

export PYTHONPATH=./

DATASET=PIADv1
SETTING=Seen
# https://huggingface.co/RayYoh/Hammer/tree/main
CPT_PATH=path/to/downloaded/model.bin

python tools/eval.py --weight $CPT_PATH --dataset $DATASET --setting $SETTING