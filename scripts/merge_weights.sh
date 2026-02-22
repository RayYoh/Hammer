#!/bin/bash

export PATH=$HOME/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

ROOT_DIR=$(pwd)

EXP="PIADv1_Seen_afford_obj_name"
SAVE_PATH="PIADv1_Unseen_afford_obj_name"
echo "Selected experiment: $EXP"
echo "Save path: $SAVE_PATH"

ckpt_dir="$ROOT_DIR/exp/$EXP/model"

# Merge the deepspeed weights to Pytorch Model
echo "<<<<<<<----->>>>>>> Preparing weights for $EXP"
cd $ckpt_dir && python zero_to_fp32.py . ../pytorch_model.bin && cd $ROOT_DIR

# Convert the Pytorch Model to Huggingface Model
echo "<<<<<<<----->>>>>>> Converting weights for $EXP"
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
python tools/merge_weights.py --weight="exp/$EXP/pytorch_model.bin" --save_path="ckpt/$SAVE_PATH"