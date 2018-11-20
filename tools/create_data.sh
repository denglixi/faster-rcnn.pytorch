#! /bin/sh
#
# create_data.sh
# Copyright (C) 2018 denglixi <denglixi@xgpd2>
#
# Distributed under terms of the MIT license.
#

#../../data/rsync.sh
./create_crosssval_dataset.py
./create_specified_imagesets.py
