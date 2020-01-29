#!/bin/bash
uname -a
#date
#env
date
CS_PATH='../lip-dataset/LIP'
LR=1e-3
WD=5e-4
BS=6
GPU_IDS=1,2,3
RESTORE_FROM='../lip-dataset/resnet101-imagenet.pth'
INPUT_SIZE='384,384'
SNAPSHOT_DIR='./snapshots_simple'
DATASET='train'
NUM_CLASSES=20
EPOCHS=50
WITH_MY_BN=0

if [[ ! -e ${SNAPSHOT_DIR} ]]; then
    mkdir -p  ${SNAPSHOT_DIR}
fi

python3 train.py --data-dir ${CS_PATH} \
       --restore-from ${RESTORE_FROM}\
       --gpu ${GPU_IDS}\
       --learning-rate ${LR}\
       --weight-decay ${WD}\
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --snapshot-dir ${SNAPSHOT_DIR}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES} \
       --epochs ${EPOCHS} \
       --with_my_bn ${WITH_MY_BN}

#python evaluate_custom.py
#--random-mirror\
#       --random-scale\
# sed -i -e 's/\r$//' run.sh