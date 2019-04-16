#!/bin/sh

# GPU usage
GPU_ID=1

# basic set

#DATASET=foodYIHmt10
#DATASET=foodexclYIHmt10_testYIH
#DATASET=food$Train$mt$Test

#DATASET=foodexclYIHmt10_testYIH
#DATASET=foodexclYIH_fineYIHfew5_testYIHfew5
#DATASET=foodexclYIH_testYIHmt10
#DATASET=foodexclYIHmt10_testYIHfew1

#DATASET=foodexclArtsmt10
DATASET=foodexclArtsmt10_testArts

#DATASET=foodexclUTownmt10_testUTownfew1
#DATASET=foodexclUTownmt10_fineUTownfew5_testUTownfew5

#DATASET=foodexclSciencemt10
#DATASET=foodexclTechChickenmt10

#DATASET=foodexclSciencemt10

#DATASET=foodexclSciencemt10

#DATASET=foodexclUTownmt10_testUTown

#DATASET=foodexclTechMixedVegmt10_testTechMixedVeg

#DATASET=foodexclYIHmt10_testYIH
IMGSET=test

#DATASET=foodAllmt10

#NET=foodres50_hierarchy_casecade_add_prob_0.5 #_casecade #{foodres50, res101, vgg16 , foodres50_hierarchy foodres50attention, foodres502fc, foodres50_hierarchy_casecade}
NET=foodres50
DATASET=schoollunch
# load weight
SESSION=42
EPOCH=35
# YIH 11545 #UTown 11407 #All 14819 #arts 13349 #science 13667
# Arts 53399
# TechChicken 13269
# TechMixed 10735
# Schoollunch 6303
CHECKPOINT=6303

# whether visulazation the results during testing
IS_VIS=false

# whether test cache which have saved in last testing
IS_TEST_CACHE=true

# whether save all detection results in images
SAVE_FOR_VIS= # blank for false


if $IS_VIS ;then
    CUDA_VISIBLE_DEVICES=$GPU_ID python gen_info_2.py --dataset $DATASET --net $NET --imgset $IMGSET \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda --vis --save_for_vis=$SAVE_FOR_VIS
elif $IS_TEST_CACHE ; then
    CUDA_VISIBLE_DEVICES=$GPU_ID python gen_info_2.py --dataset $DATASET --net $NET --imgset $IMGSET \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda --test_cache --save_for_vis=$SAVE_FOR_VIS
else
    CUDA_VISIBLE_DEVICES=$GPU_ID python gen_info_2.py --dataset $DATASET --net $NET --imgset $IMGSET \
                   --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
                   --cuda --save_for_vis=$SAVE_FOR_VIS
fi
