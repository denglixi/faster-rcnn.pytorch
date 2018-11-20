#!/bin/sh


IMAGEDIR=./userimage/
MODELDIR=./models/

#webcam_num -2 show image online -1 saveimage

# load weight
SESSION=1
EPOCH=10
CHECKPOINT=14393


# basic set
DATASET=foodAll
NET=foodres50 #{foodres50, res101, vgg16}

WEB_NUM=-2

python demo.py --net $NET \
               --dataset $DATASET \
               --image_dir $IMAGEDIR \
               --load_dir $MODELDIR \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
               --webcam_num $WEB_NUM \
               --cuda
