#!/bin/sh

# GPU usage
GPU_ID=0

# basic set
Train=All
mt=mt10
Test=

#DATASET=foodexclYIHmt10_testYIH
#DATASET=food$Train$mt$Test

#DATASET=foodexclYIH_fineYIHfew5_testYIHfew5
#DATASET=foodexclYIH_testYIHmt10
#DATASET=foodexclYIHmt10_testYIHfew1

#DATASET=foodAllmt10

DATASET=foodexclArtsmt10_testArtsfew1

NET=foodres50 #{foodres50, res101, vgg16}
# load weight
SESSION=4442
EPOCH=59
# YIH 11545 #UTown 11105 #All 14819 #arts 13349
CHECKPOINT=13349

# whether visulazation the results during testing
IS_VIS=false

# whether test cache which have saved in last testing
IS_TEST_CACHE=true

# whether save all detection results in images
SAVE_FOR_VIS=false


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
