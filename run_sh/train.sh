#!/bin/sh
# GPU usage
GPU_ID=1
WORKER_NUMBER=8

# basic set
# DATASET=foodAllmt10

#DATASET=foodexclYIHmt10
#DATASET=foodexclUTownmt10
#DATASET=foodexclArtsmt10
#DATASET=foodexclTechMixedVegmt10
#DATASET=foodexclTechChickenmt10

#DATASET=foodexclUTownmt10_fineUTownfew5
#DATASET=food_meta_Arts_train
#DATASET=foodEconomicBeeHoon

#DATASET=foodexclTechMixedVegmt10
#DATASET=foodAllmt10
DATASET=minipro_train

NET=res101 #foodres50 #{foodres50, res101, vgg16}
#NET=foodres50_hierarchy_casecade_add_prob_0.5 #_casecade #{foodres50, res101, vgg16 , foodres50_hierarchy foodres50attention, foodres502fc, foodres50_hierarchy_casecade}

SESSION=51
FIXED_LAYER=4
PRETRAIN=true
WEIGHT_FILE=imagenet #{ prefood, imagenet } only for res50
MAXEPOCHS=18
SAVE_EPOCH=3

# optimizer setting
OPTIMIZER=sgd
LEARNING_RATE=0.001
DECAY_STEP=5
IS_WARMING_UP=false
WARMING_UP_LR=0.0000001
BATCH_SIZE=1

# resume from
RESUME= # null is for false
RESUME_OPT= # null for false
RESUME_SESS_EPOCH= #null for false
CHECKSESSION=444
CHECKEPOCH=35
CHECKPOINT=11407


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
SESSION=$SESSION
FIXED_LAYER=$FIXED_LAYER
PRETRAIN=$PRETRAIN
WEIGHT_FILE=$WEIGHT_FILE
SAVE_EPOCH=$SAVE_EPOCH

----------learning rate setting----------
OPTIMIZER=$OPTIMIZER
LEARNING_RATE=$LEARNING_RATE
DECAY_STEP=$DECAY_STEP
BATCH_SIZE=$BATCH_SIZE
IS_WARMING_UP=$IS_WARMING_UP
WARMING_UP_LR=$WARMING_UP_LR
BATCH_SIZE=$BATCH_SIZE

----------resume----------
RESUME=$RESUME
RESUME_OPT = $RESUME_OPT  # null for false
RESUME_SESS_EPOCH=$RESUME_SESS_EPOCH #null for false
PRETRAIN=$PRETRAIN
DATASET=$DATASET
NET=$NET
OPTIMIZER=$OPTIMIZER
EOF

LOG=./Experiments/DetailLogs/log-`date +%Y-%m-%d-%H-%M-%S`-${DATASET}-${NET}-${SESSION}.log

# training command
if $IS_WARMING_UP ; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python sources/trainval_net.py \
                   --dataset $DATASET --net $NET \
                   --epochs $MAXEPOCHS --save_epoch=$SAVE_EPOCH \
                   --s $SESSION \
                   --o $OPTIMIZER \
                   --r=$RESUME --resume_opt=$RESUME_OPT --resume_session_epoch=$RESUME_SESS_EPOCH \
                   --fixed_layer=$FIXED_LAYER --pretrain=$PRETRAIN  --weight_file=$WEIGHT_FILE \
                   --checksession $CHECKSESSION --checkepoch $CHECKEPOCH --checkpoint $CHECKPOINT \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda --mGPUs --use_tfb --save_model \
                   --wu --wulr $WARMING_UP_LR
                       #2>&1 | tee $LOG $@
else
    CUDA_VISIBLE_DEVICES=$GPU_ID python sources/trainval_net.py \
                   --dataset $DATASET --net $NET \
                   --epochs $MAXEPOCHS --save_epoch=$SAVE_EPOCH \
                   --s $SESSION \
                   --o $OPTIMIZER \
                   --r=$RESUME --resume_opt=$RESUME_OPT --resume_session_epoch=$RESUME_SESS_EPOCH \
                   --fixed_layer=$FIXED_LAYER --pretrain=$PRETRAIN  --weight_file=$WEIGHT_FILE \
                   --checksession $CHECKSESSION --checkepoch $CHECKEPOCH --checkpoint $CHECKPOINT \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda --mGPUs --use_tfb --save_model
                       #2>&1 | tee $LOG $@
fi


