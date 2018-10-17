#!/bin/sh

SESSION=1
EPOCH=17
CHECKPOINT=2999

IMAGEDIR=./images/YIH
MODELDIR=./models/

#webcam_num -2 show image online -1 saveimage

python demo.py --net vgg16 \
               --dataset foodtechmixed \
               --image_dir $IMAGEDIR \
               --load_dir $MODELDIR \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
               --webcam_num -1 \
               --cuda
