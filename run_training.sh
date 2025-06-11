#!/bin/bash

# Define the full path to your Python executable
PYTHON_EXECUTABLE="/opt/anaconda3/bin/python"

# Path to your config file (if you are using one)
# Uncomment and adjust if you are using config.ini
# CONFIG_FILE="config.ini"

# You can easily comment out or change these defaults
DEFAULT_SAVE_DIR="./save/"
DEFAULT_LOGSTEP_TRAIN=10
DEFAULT_SAVE_MODEL="both"
DEFAULT_VAL_EVERY_N_EPOCHS=1
DEFAULT_WANDB=false # Keep as false or true
DEFAULT_WANDB_PROJECT="ML4EO"
DEFAULT_NUM_EPOCHS=100
DEFAULT_BATCH_SIZE=8
DEFAULT_DATASET="berlin"
DEFAULT_LR=0.001
DEFAULT_IMAGE_SIZE="" # No default, as it's not set in your parser
DEFAULT_SAMPLER="random"
DEFAULT_MODEL="segformer"
DEFAULT_RF_N_ESTIMATORS=100
DEFAULT_RF_MAX_DEPTH="None" # configargparse will handle "None" as None for --rf-max-depth
DEFAULT_RF_RANDOM_STATE=42
DEFAULT_RF_CLASS_WEIGHT="balanced"
DEFAULT_LAYER="layer/Berlin"
DEFAULT_USE_LAYER=true # Keep as false or true

# Override defaults with command-line arguments if provided (positional arguments)
SAVE_DIR=${1:-$DEFAULT_SAVE_DIR}
LOGSTEP_TRAIN=${2:-$DEFAULT_LOGSTEP_TRAIN}
SAVE_MODEL=${3:-$DEFAULT_SAVE_MODEL}
VAL_EVERY_N_EPOCHS=${4:-$DEFAULT_VAL_EVERY_N_EPOCHS}
WANDB=${5:-$DEFAULT_WANDB} # This will be 'true' or 'false' string
WANDB_PROJECT=${6:-$DEFAULT_WANDB_PROJECT}
NUM_EPOCHS=${7:-$DEFAULT_NUM_EPOCHS}
BATCH_SIZE=${8:-$DEFAULT_BATCH_SIZE}
DATASET=${9:-$DEFAULT_DATASET}
LR=${10:-$DEFAULT_LR}
IMAGE_SIZE=${11:-$DEFAULT_IMAGE_SIZE}
SAMPLER=${12:-$DEFAULT_SAMPLER}
MODEL=${13:-$DEFAULT_MODEL}
RF_N_ESTIMATORS=${14:-$DEFAULT_RF_N_ESTIMATORS}
RF_MAX_DEPTH=${15:-$DEFAULT_RF_MAX_DEPTH}
RF_RANDOM_STATE=${16:-$DEFAULT_RF_RANDOM_STATE}
RF_CLASS_WEIGHT=${17:-$DEFAULT_RF_CLASS_WEIGHT}
LAYER=${18:-$DEFAULT_LAYER}
USE_LAYER=${19:-$DEFAULT_USE_LAYER} # This will be 'true' or 'false' string


# Construct the command to run main.py
CMD="$PYTHON_EXECUTABLE main.py \
    --save-dir \"$SAVE_DIR\" \
    --logstep-train $LOGSTEP_TRAIN \
    --save-model $SAVE_MODEL \
    --val-every-n-epochs $VAL_EVERY_N_EPOCHS " # Note the space at the end

# Conditionally add --wandb if its value is 'true'
if [ "$WANDB" = true ]; then
    CMD="$CMD --wandb "
fi
CMD="$CMD --wandb-project \"$WANDB_PROJECT\" \
    --num_epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --dataset \"$DATASET\" \
    --lr $LR \
    --sampler \"$SAMPLER\" \
    --model \"$MODEL\" \
    --rf-n-estimators $RF_N_ESTIMATORS \
    --rf-max-depth \"$RF_MAX_DEPTH\" \
    --rf-random-state $RF_RANDOM_STATE \
    --rf-class-weight \"$RF_CLASS_WEIGHT\" \
    --layer \"$LAYER\" " # Note the space at the end

# Conditionally add --use_layer if its value is 'true'
if [ "$USE_LAYER" = true ]; then
    CMD="$CMD --use_layer "
fi

# Add image-size if it's set
if [ -n "$IMAGE_SIZE" ]; then
    CMD="$CMD --image-size $IMAGE_SIZE"
fi

echo "Running command: $CMD"
eval $CMD