#!/bin/sh

GPU_ID=0
SESSION=3
#EPOCH=5
EPOCH=40
#CHECKPOINT=1251
CHECKPOINT=2999
# --test_cache is the parameter to testing the cached result
CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset foodtechmixed --net res101 \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
