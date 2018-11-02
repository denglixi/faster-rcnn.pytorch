#!/bin/sh


IMAGEDIR=./images/techall
MODELDIR=./models/

#webcam_num -2 show image online -1 saveimage

# load weight
SESSION=10
EPOCH=69
CHECKPOINT=2999


# basic set
DATASET=foodtechmixed
NET=foodres50 #{foodres50, res101, vgg16}

WEB_NUM=-2

python demo.py --net $NET \
               --dataset $DATASET \
               --image_dir $IMAGEDIR \
               --load_dir $MODELDIR \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
               --webcam_num $WEB_NUM \
               --cuda
