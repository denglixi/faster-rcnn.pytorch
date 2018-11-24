#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 denglixi edenglixi@xgpd2>
#
# Distributed under terms of the MIT license.

"""

"""
import numpy
import os
from xml_process import parse_rec


def create_dishes(canteen):
    """create_dishes"""
    # each dish may have more than 1 image (mostly is 2).
    # construct dish-> [image1(anno) , image2(anno)]
    # dishes = [dish1, ..., dishn]
    root_path = '../data/Food/Food_{}/'.format(canteen)
    anno_path = os.path.join(root_path, 'Annotations')
    dishes = []
    dish = []

    # sort is important since listdir func do not return a ordered results.
    all_xml_fs = sorted(os.listdir(anno_path))

    for x_f in all_xml_fs:
        x_f_path = os.path.join(anno_path, x_f)
        x_f_name = x_f.split('.')[0]

        # get cls
        objs = parse_rec(x_f_path)
        cls_of_image = []
        for obj in objs:
            cls_of_image.append(obj['name'])
        cls_of_image = sorted(cls_of_image)

        # process dish & x_f_name
        if not dish:  # first image of dish
            dish = [x_f_name]
            cls_of_dish = cls_of_image
        else:
            # the same dish. TODO some samples are wrong in the following condition
            if cls_of_image == cls_of_dish or set(cls_of_image) > set(cls_of_dish) or set(cls_of_dish) > set(cls_of_image):
                dish.append(x_f_name)
            else:
                # new dish
                dishes.append(dish)
                dish = [x_f_name]
                cls_of_dish = sorted(cls_of_image)

    return dishes


def clean_validation(val_set, dishes):

    root_path = '../data/Food/Food_All/'
    imageset_path = os.path.join(root_path, 'ImageSets')
    valmt_sets_path = os.path.join(imageset_path, 'valmt10.txt')
    with open(valmt_sets_path, 'r') as f:
        valmt_names = [x.strip('\n') for x in f.readlines()]
    print(len(dishes))

    left_val = []
    for valmt_name in valmt_names:
        for dish in dishes:
            if valmt_name in dish:
                # if left_of_dish not in valmt_names
                all_in_val = True
                for img in dish:
                    if img not in valmt_name:
                        all_in_val = False
                        break
                if all_in_val:
                    left_val.append(valmt_name)
                    break

    with open("all_in_val.txt", 'w') as f:
        f.writelines([x+"\n" for x in left_val])
