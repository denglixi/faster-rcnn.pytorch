#!/bin/sh

GPU_ID=1
SESSION=1
#EPOCH=5
EPOCH=1
#CHECKPOINT=1251
CHECKPOINT=749
CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset foodtechmixed --net vgg16 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
