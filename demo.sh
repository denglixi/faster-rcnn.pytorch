#!/bin/sh

SESSION=2
EPOCH=20
CHECKPOINT=991

IMAGEDIR=./images
MODELDIR=./models/

python demo.py --net vgg16 \
               --dataset food \
               --image_dir $IMAGEDIR \
               --load_dir $MODELDIR \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
               --cuda
