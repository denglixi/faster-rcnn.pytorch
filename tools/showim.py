#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 denglixi <denglixi@xgpd9>
#
# Distributed under terms of the MIT license.

"""

"""
# import cv2

# im = cv2.imread("../data/Food/Food_YIH/JPEGImages/SGIMG_9653.jpg")
# cv2.imshow("aa", im)
# cv2.waitKey()


import matplotlib
# matplotlib.use('agg')

from PIL import Image
import matplotlib.pyplot as plt

# im = cv2.imread("./YIH/19sept_DONE380/SG/IMG_9613.JPG")
# cv2.imshow("22", im)
# cv2.waitKey()

I = Image.open("../data/Food/Food_YIH/JPEGImages/SGIMG_9613.jpg")

plt.imshow(I)
plt.show()
