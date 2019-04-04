#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 denglixi <denglixi@xgpd6>
#
# Distributed under terms of the MIT license.

"""

"""

import os

root_dir = "../data/school_lunch_dataset/"
anno_dir = os.path.join(root_dir, "Annotations")
anno_files = os.listdir(anno_dir)


train_f = open(os.path.join(root_dir, "ImageSets", "train.txt"), 'w')
val_f = open(os.path.join(root_dir, "ImageSets", "val.txt"), 'w')


for i in range(len(anno_files)):
    if i % 5 == 0:
        val_f.write(os.path.splitext(anno_files[i])[0]+'\n')
    else:
        train_f.write(os.path.splitext(anno_files[i])[0]+'\n')
