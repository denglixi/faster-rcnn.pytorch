#!/bin/sh

# GPU usage
GPU_ID=0

# basic set
Train=All
mt=mt10
Test=

#DATASET=foodexclYIHmt10_testYIH
#DATASET=food$Train$mt$Test

DATASET=foodAllmt10

NET=foodres50 #{foodres50, res101, vgg16}
# load weight
SESSION=11
EPOCH=50
# YIH 11277 #UTown 11105 #All 14389
CHECKPOINT=14391

# whether visulazation the results during testing
IS_VIS=false

# whether test cache which have saved in last testing
IS_TEST_CACHE=true

# whether save all detection results in images
SAVE_FOR_VIS=


if $IS_VIS ;then
    CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset $DATASET --net $NET \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda --vis --save_for_vis=$SAVE_FOR_VIS
elif $IS_TEST_CACHE ; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset $DATASET --net $NET \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda --test_cache --save_for_vis=$SAVE_FOR_VIS
else
    CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset $DATASET --net $NET \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda --save_for_vis=$SAVE_FOR_VIS
fi
