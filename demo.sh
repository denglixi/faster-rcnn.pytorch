#!/bin/sh

SESSION=1
EPOCH=1
CHECKPOINT=1
python demo.py --net vgg16 \
               --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
               --cuda --load_dir path/to/model/directoy
