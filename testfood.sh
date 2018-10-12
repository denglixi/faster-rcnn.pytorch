#!/bin/sh

GPU_ID=1
SESSION=2
#EPOCH=5
EPOCH=9
#CHECKPOINT=1251
CHECKPOINT=991
CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset food --net vgg16 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
