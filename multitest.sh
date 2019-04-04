#!/bin/sh

# GPU usage
GPU_ID=1

# basic set
#DATASET=foodexclYIHmt10
#DATASET=foodexclYIHmt10_fineYIH_testYIHfew1
#DATASET=foodexclYIHmt10_testYIHfew1

#DATASET=foodexclArtsmt10
#DATASET=foodexclArtsmt10_testArtsfew1


DATASET=foodexclTechChickenmt10

IMGSET=test

#DATASET=foodexclUTownmt10
#DATASET=foodexclUTownmt10_testUTownfew1

#NET=foodres50_hierarchy_casecade_add_prob_0.5 #{foodres50, res101, vgg16}
NET=foodres50attention
# load weight

# YIH 11545 #UTown 11407 #All 14819 #arts 13349 #science 13667
# Arts 53399
# TechChicken 13269
CHECKPOINT=13269

# whether visulazation the results during testing
IS_VIS=false

# whether test cache which have saved in last testing
#IS_TEST_CACHE=true
IS_TEST_CACHE=false

# whether save all detection results in images
SAVE_FOR_VIS=

#sessionlist="0 111 222 333 444 4442"
sessionlist="5"
for sess in $sessionlist
do
for i in `seq  2 3 25`
do
    EPOCH=$i
    SESSION=$sess
    if $IS_VIS ;then
        CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset $DATASET --net $NET --imgset $IMGSET \
                       --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                       --cuda --vis --save_for_vis=$SAVE_FOR_VIS
    elif $IS_TEST_CACHE ; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset $DATASET --net $NET --imgset $IMGSET \
                       --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                       --cuda --test_cache --save_for_vis=$SAVE_FOR_VIS
    else
        CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py --dataset $DATASET --net $NET --imgset $IMGSET \
                       --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                       --cuda --save_for_vis=$SAVE_FOR_VIS
    fi
done
done
