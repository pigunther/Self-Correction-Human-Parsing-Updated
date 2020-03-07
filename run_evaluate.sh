#!/bin/bash

CS_PATH='../lip-dataset/LIP'
BS=5
GPU_IDS='0'
INPUT_SIZE='384,384'
SNAPSHOT_FROM='./snapshots_simple/LIP_epoch_35.pth'
DATASET='val'
NUM_CLASSES=20
WITH_MY_BN=0

python3 eval.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES} \
       --with_my_bn ${WITH_MY_BN}

 # sed -i -e 's/\r$//' run_evaluate.sh