#!/bin/sh

GPU_ID=0

IMAGEDIR=./images/userimage/
MODELDIR=./models/
CFG_FILE=./cfgs/demo.yml

#webcam_num -2 show image online -1 saveimage

# load weight
SESSION=123
EPOCH=32
CHECKPOINT=14819


# basic set
DATASET=foodAllmt10
NET=foodres50 #{foodres50, res101, vgg16}

WEB_NUM=-2

CUDA_VISIBLE_DEVICES=$GPU_ID python demo.py --net $NET \
               --cfg $CFG_FILE \
               --dataset $DATASET \
               --image_dir $IMAGEDIR \
               --load_dir $MODELDIR \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
               --webcam_num $WEB_NUM \
               --cuda
