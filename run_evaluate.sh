#!/bin/bash

CS_PATH='../lip-dataset/LIP'
BS=6
GPU_IDS='0'
INPUT_SIZE='384,384'
SNAPSHOT_FROM='./snapshots1/LIP_epoch_49.pth'
DATASET='val'
NUM_CLASSES=20

python3 eval.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}