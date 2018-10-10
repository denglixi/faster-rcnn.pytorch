#!/bin/sh

GPU_ID=0
SESSION=1
EPOCH=6
#EPOCH=6
CHECKPOINT=1251
#CHECKPOINT=10021
CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset pascal_voc --net vgg16 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
