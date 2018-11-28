#!/bin/sh

# GPU usage
GPU_ID=1

# basic set
# DATASET=foodexclTechMixedVeg_testTechMixedVeg
#DATASET=foodexclYIHmt10
#DATASET=foodexclYIH_fineYIH_testYIHfew1
DATASET=foodexclYIHmt10_testYIHfew1
#DATASET=foodAllmt10
NET=foodres50 #{foodres50, res101, vgg16}
# load weight
SESSION=1
EPOCH=10
CHECKPOINT=11545

# whether visulazation the results during testing
IS_VIS=false

# whether test cache which have saved in last testing
#IS_TEST_CACHE=true
IS_TEST_CACHE=false

# whether save all detection results in images
SAVE_FOR_VIS=

for i in `seq 5 5 100`
do
    EPOCH=$i
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
done
