#!/bin/sh

# GPU usage
GPU_ID=1

# basic set
DATASET=foodtechmixed
NET=foodres50 #{foodres50, res101, vgg16}

# load weight
SESSION=10
EPOCH=20
CHECKPOINT=2999

# vis
IS_VIS=false

# test cache
IS_TEST_CACHE=false

if $IS_VIS ;then
    CUDA_VISIBLE_DEVICES=$GPU_ID python test_vis.py --dataset $DATASET --net $NET \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda --vis
elif $IS_TEST_CACHE ; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset $DATASET --net $NET \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda --test_cache
else
    CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset $DATASET --net $NET \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda
fi
