#!/bin/bash

# Usage: bash scripts/train.sh -d PIADv1 -p Seen -g 4 -b 64 -l 0.0001 -e 30 -n exp1

# Set the used cuda path; maybe unnecessary
export PATH=$HOME/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

ROOT_DIR=$(pwd)
TRAIN_CODE=train.py

DATASET="PIADv1"  # Options: PIADv1, PIADv2
SETTING="Seen"  # Options: Seen, Unseen_afford, Unseen_obj for PIADv2; Seen, Unseen for PIADv1
QUESTION_TYPE="afford_obj"  # Options: simple, afford, afford_obj

NUM_DEVICES=4
GLOBAL_BATCH_SIZE=64
BATCH_SIZE_PER_DEVICE=1
LR=0.0001
EPOCHS=30

while getopts "d:p:g:b:l:e:n:" opt; do
  case $opt in
    d)
      DATASET=$OPTARG
      ;;
    p)
      SETTING=$OPTARG
      ;;
    g)
      NUM_DEVICES=$OPTARG
      ;;
    b)
      GLOBAL_BATCH_SIZE=$OPTARG
      ;;
    l)
      LR=$OPTARG
      ;;
    e)
      EPOCHS=$OPTARG
      ;;
    n)
      NAME=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "$DATASET" == "PIADv2" ]; then
    STEPS_PER_EPOCH=$((37120 / GLOBAL_BATCH_SIZE))
    PRINT_FREQ=50
elif [ "$DATASET" == "PIADv1" ]; then
    STEPS_PER_EPOCH=$((12565 / GLOBAL_BATCH_SIZE))
    PRINT_FREQ=20
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

PROJECT_NAME="HAMMER_${DATASET}_$(hostname)"
EXP_NAME="${DATASET}_${SETTING}_${QUESTION_TYPE}_${NAME}"
EXP_DIR=exp/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code

echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
mkdir -p "$MODEL_DIR" "$CODE_DIR"
cp -r scripts tools src "$CODE_DIR"
cp $TRAIN_CODE "$CODE_DIR"

export PYTHONPATH=./$CODE_DIR:$PYTHONPATH
echo "Runing code in: $CODE_DIR"

if [ "$NUM_DEVICES" -gt 0 ]; then
    list=$(seq -s, 0 $((NUM_DEVICES-1)))
else
    list=""
fi
INCLUDE="localhost:$list"
echo "Using devices: $INCLUDE"

echo " =========> RUN TASK <========="

# generate random number between 20000 and 25000
MASTER_PORT=$(( ( RANDOM % 5000 )  + 20000 ))

VERSION="Qwen/Qwen2.5-VL-3B-Instruct"
LORA_R=16
LORA_ALPHA=32
DATASET_DIR="$ROOT_DIR/data"

GRAD_ACCUMULATION_STEPS=$(( GLOBAL_BATCH_SIZE / (BATCH_SIZE_PER_DEVICE * NUM_DEVICES) ))
CE_LOSS_WEIGHT=1
MASK_LOSS_WEIGHT=2


deepspeed \
  --include $INCLUDE \
  --master_port=$MASTER_PORT "$CODE_DIR"/$TRAIN_CODE \
  --version=$VERSION \
  --lora_r=$LORA_R \
  --lora_alpha=$LORA_ALPHA \
  --dataset_dir=$DATASET_DIR \
  --project_name=$PROJECT_NAME \
  --exp_name=$EXP_NAME \
  --lr=$LR \
  --epochs=$EPOCHS \
  --batch_size=$BATCH_SIZE_PER_DEVICE \
  --steps_per_epoch=$STEPS_PER_EPOCH \
  --grad_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
  --ce_loss_weight=$CE_LOSS_WEIGHT \
  --mask_loss_weight=$MASK_LOSS_WEIGHT \
  --print_freq=$PRINT_FREQ \
  --dataset=$DATASET \
  --question_type=$QUESTION_TYPE \
  --setting=$SETTING \

