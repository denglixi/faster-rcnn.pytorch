#!/bin/sh

GPU_ID=0
SESSION=1
#EPOCH=5
EPOCH=17
#CHECKPOINT=1251
CHECKPOINT=2999
CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset foodtechmixed --net vgg16 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda --test_cache
