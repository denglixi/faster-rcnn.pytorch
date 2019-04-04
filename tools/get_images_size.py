#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 denglixi <denglixi@xgpd9>
#
# Distributed under terms of the MIT license.

"""

"""
import PIL
from PIL import Image
import os

import pdb

root_dir = "../data/Food/Food_exclUTown"

img_dir = os.path.join(root_dir, 'JPEGImages')
img_files = os.listdir(img_dir)

withds = [Image.open(os.path.join(img_dir, img_f)).size[0]
          for img_f in img_files]
height = [Image.open(os.path.join(img_dir, img_f)).size[1]
          for img_f in img_files]

pdb.set_trace()
