#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 lixi <lixi@LMS-SGPU>
#
# Distributed under terms of the MIT license.

"""

"""
import os
from xml_process import parse_rec, read_xml

from PIL import Image
def travel_xml_in_path(callback_func):
    """
    travel
    callback_func : accept a function with xml object
    """
    for root, dirs, files in os.walk(root_path):
        for x_f in files:
            if os.path.splitext(x_f)[1] == '.xml':
                callback_func()



def get_all_xml_files_from_path(root_path):
    """get_all_xml_files_from_path
    recursive get all xmlfile from
    uparam root_path:
    """
    stats = {}
    log_file = open("check_box.log", 'w')
    for root, dirs, files in os.walk(root_path):
        for x_f in files:
            f_name, f_ext = os.path.splitext(x_f)
            if f_ext == '.xml':
                objs = parse_rec(os.path.join(root, x_f))
                img_path = os.path.join("/home/lixi/data/FoodDataset/Food_All/JPEGImages", f_name + '.jpg')
                width, height = Image.open(img_path).size


                #tree = read_xml(os.path.join(root, x_f))
                #size = tree.find("size")
                #width = size.find("width")
                #height = size.find("height")
                for obj in objs:
                    bbox = obj['bbox']
                    if bbox[0] == 0 or bbox[1] == 0 or bbox[2] == 0 or bbox[3] ==0:
                        log_file.write(os.path.join(root, x_f) + '\t' + "0" + '\n')
                        log_file.write(str(bbox) + '\n')
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        log_file.write(os.path.join(root, x_f) + '\t' + "1" + '\n')
                        log_file.write(str(bbox) + '\n')
                    if bbox[2] == width or bbox[3] == height:
                        log_file.write(os.path.join(root, x_f) + '\t' + "2" + '\n')
                        log_file.write(str(bbox) + '\n')

get_all_xml_files_from_path("/home/lixi/data/FoodDataset/Food_All/Annotations")
