#!/bin/sh
# GPU usage
GPU_ID=0
WORKER_NUMBER=1

# basic set
DATASET=foodAll
NET=foodres50 #{foodres50, res101, vgg16}
SESSION=1
PRETRAIN=true
MAXEPOCHS=100

# optimizer setting
OPTIMIZER=adam
LEARNING_RATE=0.01
DECAY_STEP=7
IS_WARMING_UP=false
WARMING_UP_LR=0.0000001
BATCH_SIZE=1

# resume from
RESUME= # null is for false
RESUME_OPT=1  # null for false
CHECKSESSION=7
CHECKEPOCH=14
CHECKPOINT=749


# writing the experiment detail to file
filename=./Experiments/`date +%Y-%m-%d-%H-%M-%S`-${DATASET}_${NET}_${SESSION}.log
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

LOG=./Experiments/DetailLogs/log-`date +%Y-%m-%d-%H-%M-%S`-${DATASET}-${NET}-${SESSION}.log

# training command
if $IS_WARMING_UP ; then
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
                       #2>&1 | tee $LOG $@
else
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
                   --cuda --mGPUs --use_tfb --save_model
                       #2>&1 | tee $LOG $@
fi
