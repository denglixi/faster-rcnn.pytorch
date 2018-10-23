#!/bin/sh
# GPU usage
GPU_ID=1
WORKER_NUMBER=1

# basic set
DATASET=foodtechmixed
NET=res101 #{foodres50, res101, vgg16}
SESSION=5
PRETRAIN=
MAXEPOCHS=100

# optimizer setting
OPTIMIZER=adam
LEARNING_RATE=0.001
DECAY_STEP=8
IS_WARMING_UP=false
WARMING_UP_LR=0.000001
BATCH_SIZE=4

# resume from
RESUME=true  # null is for false
RESUME_OPT=  # null for false
CHECKSESSION=5
CHECKEPOCH=29
CHECKPOINT=2999


# writing the experiment detail to file
filename="./Experiments/${DATASET}_${NET}_${SESSION}.txt"
echo "write the experiments detail to file"
cat>${filename}<<EOF
----------basic setting----------
WORKER_NUMBER = $WORKER_NUMBER
DATASET=$DATASET
NET=$NET
PRETRAIN=$PRETRAIN
MAXEPOCHS=$MAXEPOCHS

----------learning rate setting----------
OPTIMIZER=$OPTIMIZER
LEARNING_RATE=$LEARNING_RATE
DECAY_STEP=$DECAY_STEP
BATCH_SIZE=$BATCH_SIZE
IS_WARMING_UP=$IS_WARMING_UP
WARMING_UP_LR=$WARMING_UP_LR

----------resume----------
RESUME=$RESUME
RESUME_OPT = $RESUME_OPT  # null for false
PRETRAIN=$PRETRAIN
DATASET=$DATASET
NET=$NET
OPTIMIZER=$OPTIMIZER
EOF

# training command
if [ $IS_WARMING_UP ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset $DATASET --net $NET \
                   --epochs $MAXEPOCHS \
                   --s $SESSION \
                   --o $OPTIMIZER \
                   --r=$RESUME --resume_opt=$RESUME_OPT\
                   --pretrain=$PRETRAIN \
                   --checksession $CHECKSESSION --checkepoch $CHECKEPOCH --checkpoint $CHECKPOINT \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda --mGPUs --use_tfb --save_model \
                   --wu --wulr $WARMING_UP_LR
else
    CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset $DATASET --net $NET \
                   --s $SESSION \
                   --o $OPTIMIZER \
                   --r=$RESUME \
                   --epochs $MAXEPOCHS \
                   --pretrain=$PRETRAIN \
                   --checksession $CHECKSESSION --checkepoch $CHECKEPOCH --checkpoint $CHECKPOINT \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda --mGPUs --use_tfb --save_model
fi
