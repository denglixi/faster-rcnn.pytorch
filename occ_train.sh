#!/bin/sh
GPU_ID=0,1
WORKER_NUMBER=2
LEARNING_RATE=0.001
DECAY_STEP=5
BATCH_SIZE=16
# resume from
RESUME=  #null is for false
#session is training session
SESSION=1  # 2 is for food_YIH
CHECKSESSION=1
EPOCH=7
CHECKPOINT=10021
CUDA_VISIBLE_DEVICES=$GPU_ID
python trainval_net.py \
                   --dataset foodtechmixed --net vgg16 \
                   --s $SESSION \
                   --r=$RESUME \
                   --checksession $CHECKSESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --epochs 100000 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda --mGPUs
